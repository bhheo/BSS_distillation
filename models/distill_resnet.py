'''ResNet for knowledge distillation
A structure saves mid-layer response
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class Distill_ResNet_Simple(nn.Module):
    def __init__(self, ori_net):
        super(Distill_ResNet_Simple, self).__init__()

        self.conv1 = ori_net.conv1
        self.bn1 = ori_net.bn1
        self.layer1 = ori_net.layer1
        self.layer2 = ori_net.layer2
        self.layer3 = ori_net.layer3
        self.linear = ori_net.linear

    def forward(self, x):
        self.res0 = F.relu(self.bn1(self.conv1(x)))

        self.res1 = self.layer1(self.res0)
        self.res2 = self.layer2(self.res1)
        self.res3 = self.layer3(self.res2)

        out = F.avg_pool2d(self.res3, 8)
        out = out.view(out.size(0), -1)
        self.out = self.linear(out)
        return self.out

