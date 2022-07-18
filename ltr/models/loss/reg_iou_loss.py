import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


eps = 1e-6
class IOULoss(nn.Module):

    def __init__(self, mode='iou', average_size=True):
        super(IOULoss, self).__init__()
        self.iou_mode = mode 
        self.average_size =average_size 

    def forward(self, pred, label, weight=None):
        pred_ = pred 
        label_ = label  

        left = torch.cat([pred_[:, 0:1, :, :], label_[:, 0:1, :, :]], dim=1)
        right = torch.cat([pred_[:, 2:3, :, :], label_[:, 2:3, :, :]], dim=1)
        top = torch.cat([pred_[:, 1:2, :, :], label_[:, 1:2, :, :]], dim=1)
        bottom = torch.cat([pred_[:, 3:4, :, :], label_[:, 3:4, :, :]], dim=1)

        i_left = left.min(dim=1, keepdim=True)[0]
        i_right = right.min(dim=1, keepdim=True)[0]
        i_top = top.min(dim=1, keepdim=True)[0]
        i_bottom = bottom.min(dim=1, keepdim=True)[0]
        i_area = F.relu(i_right + i_left) * F.relu(i_bottom + i_top)

        pred_area = (pred_[:, 2:3, :, :] + pred_[:, 0:1, :, :]) * (pred_[:, 3:4, :, :] + pred_[:, 1:2, :, :])
        label_area = (label_[:, 2:3, :, :] + label_[:, 0:1, :, :]) * (label_[:, 3:4, :, :] + label_[:, 1:2, :, :])
        u_area = pred_area + label_area - i_area

        iou = (i_area + 1) / (u_area + 1)

        if self.iou_mode == 'iou':
            losses = -torch.log(iou)
        elif self.iou_mode == 'giou':
            e_left = left.max(dim=1, keepdim=True)[0]
            e_right = right.max(dim=1, keepdim=True)[0]
            e_top = top.max(dim=1, keepdim=True)[0]
            e_bottom = bottom.max(dim=1, keepdim=True)[0]
            enclose_area = (e_left + e_right) * (e_top + e_bottom)
            giou = iou - (enclose_area - u_area) / (enclose_area + eps)
            losses = -torch.log((giou+1)/2)
        else:
            raise ValueError('Unknown iou type: {}'.format(self.mode))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class DistanceLoss(nn.Module):

    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, pred, label, use_mask):
        distance = torch.abs(pred - label)
        distance = distance * use_mask 

        num_samples = torch.sum(torch.sum(use_mask, dim=-1, keepdim=True), dim=-2, keepdim=True)
        sum_samples_loss = torch.mean(torch.sum(torch.sum(distance * use_mask, dim=-1, keepdim=True), dim=-2, keepdim=True), dim=-3, keepdim=True)
        valid_flag = (num_samples > 0).float()
        return (sum_samples_loss / (num_samples + eps)).sum() / valid_flag.sum()
