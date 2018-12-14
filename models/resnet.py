'''ResNet in PyTorch.

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable

class ZeroPadBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ZeroPadBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=1, stride=stride)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += F.pad(self.shortcut(x), (0, 0, 0, 0, 0, out.size()[1] - x.size()[1]), 'constant', 0)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        multiplier = 1
        self.in_planes = multiplier*16

        self.conv1 = nn.Conv2d(3, multiplier*16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(multiplier*16)
        self.layer1 = self._make_layer(block, multiplier*16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, multiplier*32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, multiplier*64, num_blocks[2], stride=2)
        self.linear = nn.Linear(multiplier*64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def BN_version_fix(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True
            m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    return net

def ResNet8():
    return ResNet(ZeroPadBlock, [1,1,1])

def ResNet14():
    return ResNet(ZeroPadBlock, [2,2,2])

def ResNet20():
    return ResNet(ZeroPadBlock, [3,3,3])

def ResNet26():
    return ResNet(ZeroPadBlock, [4,4,4])


