import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict 


class EnRefiner(nn.Module):

    def __init__(self, block_num=1, nhead=4, layer=['layer2', 'layer3'], feature_dim=[512, 1024], inner_dim=[512, 1024]):
        super(EnRefiner, self).__init__()
        for i in range(len(layer)):
            l = layer[i]
            setattr(self, 'encoder_{}'.format(l),
            nn.TransformerEncoder(num_layers=block_num, norm=None,
            encoder_layer=nn.TransformerEncoderLayer(feature_dim[i], nhead, inner_dim[i])))

    def forward(self, feat):
        out = OrderedDict()
        for key in feat.keys():
            refine_feat = getattr(self, 'encoder_{}'.format(key))(feat[key])
            out[key] = refine_feat 
        return out 


def en_refiner50():
    return EnRefiner(block_num=1, nhead=4, layer=['layer2', 'layer3'], 
                     feature_dim=[512, 1024], inner_dim=[512, 1024])

def en_refiner18():
    return EnRefiner(block_num=1, nhead=4, layer=['layer2', 'layer3'], 
                     feature_dim=[128, 256], inner_dim=[256, 512])
