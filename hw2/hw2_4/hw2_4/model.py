import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # TODO

    def forward(self, x):
        # TODO
        return out

    def name(self):
        return "ConvNet"

class Fully(nn.Module):
    def __init__(self):
        super(Fully, self).__init__()
        # TODO

    def forward(self, x):
        x = x.view(x.size(0),-1) # flatten input tensor
        # TODO
        return out

    def name(self):
        return "Fully"

