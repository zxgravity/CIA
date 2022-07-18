import torch
import torch.nn.functional as F
import torchvision as tv 
import random

import pdb 



def compute_locations(w, h, stride, device):

    shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2

    return locations 

def ltrb2xyxy(ltrb, stride, device=None):

    if device is None:
        device = ltrb.device 
    h, w = ltrb.shape[-2:]
    locations = compute_locations(w, h, stride, device)
    locations = locations.permute(1, 0).contiguous().view(1, 2, h, w) 
    x1 = locations[:, 0:1, :, :] - ltrb[:, 0:1, :, :]
    x2 = locations[:, 0:1, :, :] + ltrb[:, 2:3, :, :]
    y1 = locations[:, 1:2, :, :] - ltrb[:, 1:2, :, :]
    y2 = locations[:, 1:2, :, :] + ltrb[:, 3:4, :, :]

    xyxy = torch.cat([x1, y1, x2, y2], dim=1)

    return xyxy

def xywh2xyxy(xywh, device=None):

    if device is None:
        device = xywh.device 
    xyxy = xywh.clone().to(device)
    xyxy[:, 2:4] = xyxy[:, 0:2] + xyxy[:, 2:4]

    return xyxy 

def cwh2xyxy(cwh, device=None):

    if device is None:
        device = cwh.device
    xyxy = cwh.clone().to(device)
    xyxy[:, 0:2] = xyxy[:, 0:2] - xyxy[:, 2:4] / 2
    xyxy[:, 2:4] = xyxy[:, 0:2] + xyxy[:, 2:4]

    return xyxy

def xyxy2cwh(xyxy, device=None):

    if device is None:
        device = xyxy.device 
    cwh = xyxy.clone().to(device)
    cwh[:, 0:2] = (xyxy[:, 0:2] + xyxy[:, 2:4]) / 2
    cwh[:, 2:4] = xyxy[:, 2:4] - xyxy[:, 0:2]

    return cwh 

def xywh2cwh(xywh, device=None):

    if device is None:
        device = xywh.device 
    cwh = xywh.clone().to(device)
    cwh[:, 0:2] = cwh[:, 0:2] - cwh[:, 2:4] / 2

    return cwh 

def calculate_iou(box1, box2, type='xyxy'): 
    '''
    box1 : [N, 4]
    type : can be set as xyxy, ltrb, xyhw, chw
    '''
    if type == 'xyxy':
        w1 = box1[:, 2] - box1[:, 0]
        h1 = box1[:, 3] - box1[:, 1]
        w2 = box2[:, 2] - box2[:, 0]
        h2 = box2[:, 3] - box2[:, 1]

        x1_i = torch.max(box1[:, 0], box2[:, 0])
        x2_i = torch.min(box1[:, 2], box2[:, 2])
        y1_i = torch.max(box1[:, 1], box2[:, 1])
        y2_i = torch.min(box1[:, 3], box2[:, 3])

        wi = F.relu(x2_i - x1_i)
        hi = F.relu(y2_i - y1_i)

        area_i = wi * hi 
        area_u = w1 * h1 + w2 * h2 - area_i 

        iou = area_i / area_u 

    return iou 




def bb2roi(bb, bb_format='xywh', device=None):
    xyxy = eval('{}2xyxy'.format(bb_format))(bb, device)
    batch_size = bb.shape[0]
    batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(xyxy.device)
    rois = torch.cat((batch_index, xyxy), dim=1)

    return rois 




def corr(weight, feat):
    
    batch = feat.shape[0]
    weight = weight.view(-1, *weight.shape[-3:])
    feat = feat.view(1, -1, *feat.shape[-2:])
    out = F.conv2d(feat, weight, groups=batch)

    return out.view(batch, -1, *out.shape[-2:])

def dw_corr(weight, feat):

    batch = feat.shape[0]
    weight = weight.view(-1, 1, *weight.shape[-2:])
    feat = feat.view(1, -1, *feat.shape[-2:])
    out = F.conv2d(feat ,weight, groups=feat.shape[1], padding=(weight.shape[-1]-1)//2)

    return out.view(batch, -1, *out.shape[-2:])

def deform_dw_corr(weight, feat, offset):

    batch = feat.shape[0]
    deform_out = [tv.ops.deform_conv2d(feat[i:i+1], offset[i:i+1], weight[i:i+1].permute(1,0,2,3), padding=weight.shape[-1]//2) for i in range(batch)]
    return torch.cat(deform_out, dim=0)

def point2point_corr(weight, feat):

    batch, channel, h, w = feat.shape 
    weight = weight.permute(0, 2, 3, 1).contiguous()
    weight = weight.view(-1, channel, 1, 1)
    feat = feat.view(1, -1, *feat.shape[-2:])
    p2p_corr = F.conv2d(feat, weight, groups=batch)
    return p2p_corr.view(batch, int(h*w), *feat.shape[-2:])







def get_noise_box(box, sample_num=5, shift_ratio=0.5, scale_ratio=0.2):
    '''
    box : [xc, yc, w, h]    [N, 4] Tensor
    '''
    batch = box.shape[0]
    box_all = []
    for i in range(batch):
        box_all.append(box[i])
        for j in range(sample_num):
            shift_x = random.uniform(-shift_ratio, shift_ratio) * box[i, 2]
            shift_y = random.uniform(-shift_ratio, shift_ratio) * box[i, 3]
            xc = box[i, 0] + shift_x 
            yc = box[i, 1] + shift_y 
            scale_x = random.uniform(1-scale_ratio, 1+scale_ratio)
            scale_y = random.uniform(1-scale_ratio, 1+scale_ratio)
            w = box[i, 2] * scale_x 
            h = box[i, 3] * scale_y
            box_all.append([xc, yc, w, h])
    noise_boxes = torch.tensor(box_all, dtype=torch.float64, device=box.device)
    return noise_boxes 






def atss_asign(locations, gt, k=9, anchor_sz=288/5):
    '''
    locations: [[x1, x2, ..., xn],[y1, y2, ..., yn]]
    gt       : [N, 4], xyxy form 
    k        : number of selected anchors for calculating iou threshold
    anchor_sz: anchor size for every anchor point
    '''
    batch = gt.shape[0]
    gt_center = (gt[:, 0:2] + gt[:, 2:4]) / 2
    gt_center = gt_center[:, None, :]
    locations = locations[None, :, :]
    d_center = torch.sum((locations - gt_center)**2, dim=-1)
    val, index = d_center.sort(dim=-1)
    select_index = index[:, 0:k]
    k_candi_list = []
    for bi in range(batch):
        k_candi_list.append(torch.index_select(locations, 1, select_index[bi]))
    k_candi = torch.cat(k_candi_list, dim=0)
    candi_wh = torch.tensor([anchor_sz, anchor_sz], dtype=torch.float32, device=locations.device)
    candi_wh = candi_wh[None, None, :].repeat(batch, k, 1)
    k_candi = torch.cat([k_candi, candi_wh], dim=-1).view(-1, 4)
    pdb.set_trace()
    k_candi = cwh2xyxy(k_candi)
    gt_repeat = gt.unsqueeze(1).repeat(1, k, 1).view(-1, 4)
    candi_iou = calculate_iou(k_candi, gt_repeat).view(batch, k, 1)
    return 





if __name__ == '__main__':
    gt = torch.tensor([[27, 25, 37, 39],[24, 29, 35, 40]], dtype=torch.float32)
    locations = compute_locations(8, 8, 8, gt.device)
    mask = atss_asign(locations, gt, anchor_sz=64/5)

