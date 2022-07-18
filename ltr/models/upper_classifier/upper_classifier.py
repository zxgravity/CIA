import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from .transformer import *
import pdb 


class FCClassifier(nn.Module):

    def __init__(self, class_num=70, in_channel=2048, post_proj_channel=256, feature_stride=32):
        super().__init__()
        self.prroi_pool = PrRoIPool2D(1, 1, 1/feature_stride)
        self.proj_fc = nn.Sequential(nn.Linear(in_channel, in_channel*2),
                                     nn.ReLU(inplace=True))
        self.clf_fc = nn.Linear(in_channel*2, class_num)

        self.upper_clf_channel = in_channel
        self.lower_clf_channel = in_channel
        if not self.upper_clf_channel + self.lower_clf_channel == self.clf_fc.weight.shape[1]:
            raise ValueError('The sum of upper and lower channel numbers is not right.')
        self.cls_vectors = self.clf_fc.weight[:, self.upper_clf_channel:].cuda().detach()
        self.proj_post = nn.Linear(self.lower_clf_channel, post_proj_channel)

        transformer_encoder_layer = TransformerEncoderLayer(d_model=post_proj_channel, nhead=4)
        self.transformer_encoder = TransformerEncoder(encoder_layer=transformer_encoder_layer, num_layers=3, norm=None)

    def forward(self, feat, bb):
        pool_bb = bb.clone()
        pool_bb[:, 2:4] = pool_bb[:, 0:2] + pool_bb[:, 2:4]
        batch_index = torch.arange(pool_bb.shape[0], dtype=torch.float32).reshape(-1, 1).to(pool_bb.device)
        rois = torch.cat([batch_index, pool_bb], dim=1)

        feat_rois = self.prroi_pool(feat, rois)
        feat_rois = feat_rois.squeeze()
        feat_cls = self.proj_fc(feat_rois)
        cls_pred = self.clf_fc(feat_cls)

        cls_vectors = self.clf_fc.weight[:, self.upper_clf_channel:]
        self.cls_vectors = cls_vectors 

        return cls_pred, cls_vectors

    def classify(self, feat, bb):
        pool_bb = bb.clone()
        pool_bb[:, 2:4] = pool_bb[:, 0:2] + pool_bb[:, 2:4]
        batch_index = torch.arange(pool_bb.shape[0], dtype=torch.float32).reshape(-1, 1).to(pool_bb.device)
        rois = torch.cat([batch_index, pool_bb], dim=1)

        feat_rois = self.prroi_pool(feat, rois)
        feat_rois = feat_rois.squeeze()
        feat_cls = self.proj_fc(feat_rois)
        cls_pred = self.clf_fc(feat_cls)

        return cls_pred 

    def semantic_trans(self, feat, training=True):
        feat_reshape = feat.reshape(*feat.shape[0:2], -1).permute(2, 0, 1)
        cls_vectors_proj = self.proj_post(self.cls_vectors)
        cls_vectors_repeat = cls_vectors_proj.unsqueeze(1).expand(-1, feat.shape[0], -1)
        feat_trans = torch.cat([feat_reshape, cls_vectors_repeat], dim=0)
        out_trans = self.transformer_encoder(feat_trans, training=training)
        out = out_trans[0:feat_reshape.shape[0], :, :].permute(1, 2, 0)
        out = out.reshape(*feat.shape) + feat 
        return out


class FCClassifier_v2(nn.Module):

    def __init__(self, class_num=70, in_channel=2048, post_proj_channel=256, feature_stride=32):
        super().__init__()
        self.prroi_pool = PrRoIPool2D(1, 1, 1/feature_stride)
        self.proj_fc = nn.Sequential(nn.Linear(in_channel, in_channel*2),
                                     nn.ReLU(inplace=True))
        self.clf_fc = nn.Linear(in_channel*2, class_num)

        self.upper_clf_channel = in_channel
        self.lower_clf_channel = in_channel
        if not self.upper_clf_channel + self.lower_clf_channel == self.clf_fc.weight.shape[1]:
            raise ValueError('The sum of upper and lower channel numbers is not right.')
        self.cls_vectors = self.clf_fc.weight[:, self.upper_clf_channel:].cuda().detach()
        self.proj_post = nn.Linear(self.lower_clf_channel, post_proj_channel)

        transformer_encoder_layer = TransformerEncoderLayer_v2(d_model=post_proj_channel, nhead=4)
        self.transformer_encoder = TransformerEncoder_v2(encoder_layer=transformer_encoder_layer, num_layers=3, norm=None)

    def forward(self, feat, bb):
        pool_bb = bb.clone()
        pool_bb[:, 2:4] = pool_bb[:, 0:2] + pool_bb[:, 2:4]
        batch_index = torch.arange(pool_bb.shape[0], dtype=torch.float32).reshape(-1, 1).to(pool_bb.device)
        rois = torch.cat([batch_index, pool_bb], dim=1)

        feat_rois = self.prroi_pool(feat, rois)
        feat_rois = feat_rois.squeeze()
        feat_cls = self.proj_fc(feat_rois)
        cls_pred = self.clf_fc(feat_cls)

        cls_vectors = self.clf_fc.weight[:, self.upper_clf_channel:]
        self.cls_vectors = cls_vectors 

        return cls_pred, cls_vectors 
        # return cls_pred, cls_vectors, feat_rois   # for cls visualization

    def classify(self, feat, bb):
        pool_bb = bb.clone()
        pool_bb[:, 2:4] = pool_bb[:, 0:2] + pool_bb[:, 2:4]
        batch_index = torch.arange(pool_bb.shape[0], dtype=torch.float32).reshape(-1, 1).to(pool_bb.device)
        rois = torch.cat([batch_index, pool_bb], dim=1)

        feat_rois = self.prroi_pool(feat, rois)
        feat_rois = feat_rois.squeeze()
        feat_cls = self.proj_fc(feat_rois)
        cls_pred = self.clf_fc(feat_cls)

        return cls_pred 

    def semantic_trans(self, feat, training=True):
        feat_reshape = feat.reshape(*feat.shape[0:2], -1).permute(2, 0, 1)
        cls_vectors_proj = self.proj_post(self.cls_vectors)
        cls_vectors_repeat = cls_vectors_proj.unsqueeze(1).expand(-1, feat.shape[0], -1)
        out_trans = self.transformer_encoder(feat_reshape, cls_vectors_repeat, cls_vectors_repeat, training=training)
        out = out_trans.permute(1, 2, 0)
        out = out.reshape(*feat.shape) + feat 
        return out


