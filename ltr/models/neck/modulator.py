import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ltr.models.utils import corr 
from ltr.models.layers.normalization import InstanceL2Norm 
from collections import OrderedDict 

import pdb


class Modulator(nn.Module):
    def __init__(self, alpha=-1, alpha_mode='fixed'):
        super().__init__()
        self.alpha_mode = alpha_mode 
        if alpha_mode == 'learned':
            self.alpha_factor = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        else:
            self.alpha = alpha 

    def forward(self, input, modulation):
        denorm = torch.norm(input, p=2, dim=1, keepdim=True).clamp_min(1e-12)
        if self.alpha_mode == 'learned':
            self.alpha = torch.exp(self.alpha_factor)

        if self.alpha <= 0:
            output = input + modulation
            output = F.normalize(output, dim=1)
            output = output * denorm
        else:
            denorm_change = denorm / self.alpha
            input = input / denorm_change
            output = input + modulation
            output = F.normalize(output, dim=1)
            output = output * denorm

        return output


class Modulator_v2(nn.Module):
    def __init__(self, in_channels=256, alpha=-1, alpha_mode='fixed'):
        super().__init__()
        self.alpha_mode = alpha_mode 
        if alpha_mode == 'learned':
            self.alpha_factor = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        else:
            self.alpha = alpha 

        self.proj = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=True),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1))
        for m in self.proj.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels 
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, modulation):
        denorm = torch.norm(input, p=2, dim=1, keepdim=True).clamp_min(1e-12)
        if self.alpha_mode == 'learned':
            self.alpha = torch.exp(self.alpha_factor)

        if self.alpha <= 0:
            output = input + modulation
            output = F.normalize(output, dim=1)
            output = output * denorm
        else:
            denorm_change = denorm / self.alpha
            input = input / denorm_change
            modulation = self.proj(modulation)
            output = input + modulation
            output = F.normalize(output, dim=1)
            output = output * denorm

        return output


class Modulator_v3(nn.Module):
    def __init__(self, in_channels=256): #, alpha=-1, alpha_mode='fixed'):
        super().__init__()
        # self.alpha_mode = alpha_mode 
        # if alpha_mode == 'learned':
        #     self.alpha_factor = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        # else:
        #     self.alpha = alpha 

        self.proj = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=True),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1))
        for m in self.proj.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels 
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, modulation):
        modulation = self.proj(modulation)
        output = input * modulation 
        return output
