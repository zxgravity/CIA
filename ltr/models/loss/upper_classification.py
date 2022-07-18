import torch
import torch.nn as nn

import pdb



class UpperLoss(nn.Module):

    def __init__(self, crt):
        super().__init__()
        self.crt = crt 

    def forward(self, input, target):
        target = torch.tensor(target, dtype=torch.long).to(input.device)
        total_num = target.shape[0]
        invalid_index = torch.where(target == -1)[0]
        target[invalid_index] = 0
        mask = torch.ones(total_num).to(input.device)
        mask[invalid_index] = 0
        invalid_num = invalid_index.shape[0]
        valid_num = total_num - invalid_num

        # valid_index = torch.where(target > -1)[0]
        # valid_num = valid_index.shape[0]
        # valid_target = torch.index_select(target, 0, valid_index)
        # valid_input = torch.index_select(input, 0, valid_index)
        if valid_num == 0:
            loss = torch.tensor(0, dtype=torch.float).to(input.device)
        else:
            loss = self.crt(input, target)
            loss = (loss * mask).sum() / valid_num
        return loss 

