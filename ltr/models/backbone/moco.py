import numpy as np 
import torch 
import torch.nn as nn
from .resnet import resnet18, resnet50
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from ltr.models.utils import bb2roi
from ltr.models.neck import Modulator, Modulator_v2, Modulator_v3

import pdb 


class Projector(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels=None, stride=1):
        super().__init__()

        self.stride = stride 
        if out_channels is None:
            out_channels = in_channels 
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, bias=True, 
                               padding=1, stride=stride)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.act = nn.ReLU()

    def forward(self, input, padding=1, stride=1):
        weights = self.conv1.weight
        bias = self.conv1.bias
        output = nn.functional.conv2d(input, weights, bias, padding=padding, stride=stride)
        output = self.act(output)
        output = self.conv2(output)
        return output


class MoCo(nn.Module):
    def __init__(self, encoder_q, encoder_k,
                 pool_size=3, pool_stride=16, dim=128,
                 K=65536, m=0.999, T=0.07, alpha=-1, alpha_mode='fixed',
                 mlp=True, moco_layer='layer3', modulator='v2'):
        """
        # instance modulator v2 
        dim: feature dimension
        K: queue size
        m: moco momentum of updating key encoder
        T: softmax temperature
        """
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        self.moco_layer = moco_layer

        # create the encoders
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        if mlp:
            try:
                # for resnet50
                dim_mlp = self.encoder_q.layer3[-1].conv3.weight.shape[0]
            except:
                # for resnet18
                dim_mlp = self.encoder_q.layer3[-1].conv2.weight.shape[0]
            self.encoder_q.proj = Projector(in_channels=dim_mlp, kernel_size=3, stride=1)
            self.encoder_k.proj = Projector(in_channels=dim_mlp, kernel_size=3, stride=1)

        self.prroi_pool = PrRoIPool2D(pool_size, pool_size, 1/pool_stride)

        if modulator == 'v1':
            self.modulator = Modulator(alpha=alpha, alpha_mode=alpha_mode)
        elif modulator == 'v2':
            self.modulator = Modulator_v2(in_channels=dim_mlp, alpha=alpha, alpha_mode=alpha_mode)
        elif modulator == 'v3':
            self.modulator = Modulator_v3(in_channels=dim_mlp)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        # replace the keys at ptr
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffle index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, bb_q, bb_k, layers):
        """
        Input:
            im_q: a batch of query images (NxBx3xHxW)
            im_k: a batch of key images (NxBx3xHxW)
        """

        # MOCO sample extraction
        num_imgs, batch_size = im_q.shape[0:2]
        # compute query features
        im_q = im_q.permute(1, 0, 2, 3, 4).reshape(-1, *im_q.shape[-3:])
        feat_q = self.encoder_q(im_q, layers)
        feat_q_moco = feat_q[self.moco_layer] # feat_q_moco: (BxN)xCxHxW
        bb_q_ = bb_q.clone().permute(1, 0, 2).reshape(-1, 4)
        rois_q = bb2roi(bb_q_)

        feat_q_proj = self.encoder_q.proj(feat_q_moco)
        feat_q_proj = nn.functional.normalize(feat_q_proj, dim=1)

        q = self.prroi_pool(feat_q_moco, rois_q)
        q = self.encoder_q.proj(q, padding=0).squeeze()
        q = nn.functional.normalize(q, dim=1)
        # q = q.reshape(batch_size, num_imgs, *q.shape[-3:])

        with torch.no_grad():
            self._momentum_update_key_encoder()

            im_k = im_k.permute(1, 0, 2, 3, 4).reshape(-1, *im_k.shape[-3:])
            feat_k = self.encoder_k(im_k, layers)
            feat_k_moco = feat_k[self.moco_layer] # feat_k_moco: (BxN)xCxHxW
            bb_k_ = bb_k.clone().permute(1, 0, 2).reshape(-1, 4)
            rois_k = bb2roi(bb_k_)

            # feat_k_proj = self.encoder_k.proj(feat_k_moco)
            # feat_k_proj = nn.functional.normalize(feat_k_proj, dim=1)

            k = self.prroi_pool(feat_k_moco, rois_k)
            k = self.encoder_k.proj(k, padding=0).squeeze()
            k = nn.functional.normalize(k, dim=1)
            # k = k.reshape(batch_size, num_imgs, *k.shape[-3:])

        q_temperature = q / self.T

        # Instance modulator
        feat_q_proj = feat_q_proj.detach()
        feat_q_ins = self.modulator(feat_q_moco, feat_q_proj)
        feat_q[self.moco_layer+'_ins'] = feat_q_ins

        return feat_q, q_temperature, k

    def extract_backbone_feature(self, im, layers):
        num_imgs, batch_size = im.shape[0:2]
        im = im.reshape(-1, *im.shape[-3:])
        features = self.encoder_q(im, layers)

        feat_q_proj = self.encoder_q.proj(features[self.moco_layer])
        feat_q_proj = nn.functional.normalize(feat_q_proj, dim=1)

        # Instance modulator
        feat_q_ins = self.modulator(features[self.moco_layer], feat_q_proj)
        features[self.moco_layer+'_ins'] = feat_q_ins

        features['ins_proj'] = feat_q_proj 

        return features

def moco_resnet18(output_layers=None, pretrained=False, frozen_layers=[]):
    encoder_q = resnet18(output_layers=output_layers, pretrained=pretrained, frozen_layers=frozen_layers)
    encoder_k = resnet18(output_layers=output_layers, pretrained=pretrained)

    model = MoCo(encoder_q, encoder_k, alpha=-1, alpha_mode='learned')

    return model

def moco_resnet50(output_layers=None, pretrained=False, frozen_layers=[]):
    encoder_q = resnet50(output_layers=output_layers, pretrained=pretrained, frozen_layers=frozen_layers)
    encoder_k = resnet50(output_layers=output_layers, pretrained=pretrained)

    model = MoCo(encoder_q, encoder_k, alpha=-1, alpha_mode='learned')

    return model


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors
    *** Warning ***: torch,distributed.all_gather has no gradient. 
    """
    tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output 




