import torch
import torch.nn as nn 


class InfoNCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, label):
        input = torch.exp(input)
        numerator = torch.sum(input * label)
        denominator = torch.sum(input)
        loss = -torch.log(numerator / denominator)

        return loss 
