import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmengine.model import BaseModule

def upsample(x, size):
    if x.shape[-2] != size[0] or x.shape[-1] != size[1]:
        return F.interpolate(x, size, mode='bilinear', align_corners=True)
    else:
        return x

def bi_interaction(x_h, x_l):
    sizeH = (int(x_h.shape[-2]), int(x_h.shape[-1]))
    sizeL = (int(x_l.shape[-2]), int(x_l.shape[-1]))
    o_h = x_h + upsample(x_l, sizeH)
    o_l = x_l + upsample(x_h, sizeL)
    return o_h, o_l

def tr_interaction(x1, x2, x3):
    s1 = (int(x1.shape[-2]), int(x1.shape[-1]))
    s2 = (int(x2.shape[-2]), int(x2.shape[-1]))
    s3 = (int(x3.shape[-2]), int(x3.shape[-1]))
    o1 = x1 + upsample(x2, s1) + upsample(x3, s1)
    o2 = x2 + upsample(x1, s2) + upsample(x3, s2)
    o3 = x3 + upsample(x2, s3) + upsample(x1, s3)
    return o1, o2, o3

class ConvBNReLU3x3(BaseModule):
    def __init__(self, c_in, c_out, stride, deploy=False, init_cfg=None):
        super(ConvBNReLU3x3, self).__init__(init_cfg)
        if deploy:
            self.conv = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=True)
            self.bn = None
        else:
            self.conv = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
            self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x

class BaseNet(BaseModule):
    def __init__(self, layers, channels, deploy=False, init_cfg=None):
        super(BaseNet, self).__init__(init_cfg)
        self.layers = layers
        assert len(self.layers) == 5
        self.channels = channels
        assert len(self.channels) == 5
        self.strides = (2, 2, 2, 2, 1)
        self.stages = nn.ModuleList()
        c_in = 3
        for l, c, s in zip(self.layers, self.channels, self.strides):
            self.stages.append(self._make_stage(c_in, c, l, s, deploy))
            c_in = c

    @staticmethod
    def _make_stage(c_in, c_out, numlayer, stride, deploy):
        layers = []
        for i in range(numlayer):
            layers.append(ConvBNReLU3x3(c_in if i == 0 else c_out, c_out, stride if i == 0 else 1, deploy))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        outputs = []
        for s in self.stages:
            out = s(out)
            outputs.append(out)
        return outputs

@MODELS.register_module()
class LPSNet(BaseModule):
    def __init__(self, depth, width, resolution, deploy=False, init_cfg=None):
        super(LPSNet, self).__init__(init_cfg)
        self.depth = depth
        assert len(self.depth) == 5
        self.width = width
        assert len(self.width) == 5
        self.resolution = resolution
        self.deploy = deploy

        self.resolution_filter = [r for r in resolution if r > 0]
        self.resolution_sorted = sorted(self.resolution_filter, reverse=True)
        self.num_paths = len(self.resolution_sorted)
        assert self.num_paths in [1, 2, 3], 'Only support 1, 2, or 3 paths'

        if self.num_paths == 1:
            self.net = BaseNet(self.depth, self.width, deploy)
        elif self.num_paths == 2:
            self.netH = BaseNet(self.depth, self.width, deploy)
            self.netL = BaseNet(self.depth, self.width, deploy)
        elif self.num_paths == 3:
            self.net1 = BaseNet(self.depth, self.width, deploy)
            self.net2 = BaseNet(self.depth, self.width, deploy)
            self.net3 = BaseNet(self.depth, self.width, deploy)
        else:
            raise NotImplementedError

    def _preprocess_input(self, x):
        r_list = self.resolution_sorted
        x_list = [upsample(x, (int(x.shape[-2] * r), int(x.shape[-1] * r))) for r in r_list]
        return x_list

    def forward(self, x):
        if self.num_paths == 1:
            x_processed = self._preprocess_input(x)[0]
            outs = self.net(x_processed)
            return [outs[-1]]  # Return only the last feature
        elif self.num_paths == 2:
            xh, xl = self._preprocess_input(x)
            xh, xl = self.netH.stages[0](xh), self.netL.stages[0](xl)
            xh, xl = self.netH.stages[1](xh), self.netL.stages[1](xl)
            xh, xl = self.netH.stages[2](xh), self.netL.stages[2](xl)
            xh, xl = bi_interaction(xh, xl)
            xh, xl = self.netH.stages[3](xh), self.netL.stages[3](xl)
            xh, xl = bi_interaction(xh, xl)
            xh, xl = self.netH.stages[4](xh), self.netL.stages[4](xl)
            x_cat = torch.cat([xh, upsample(xl, (int(xh.shape[-2]), int(xh.shape[-1])))], dim=1)
            return [x_cat]  # Return the concatenated feature
        elif self.num_paths == 3:
            x1, x2, x3 = self._preprocess_input(x)
            x1, x2, x3 = self.net1.stages[0](x1), self.net2.stages[0](x2), self.net3.stages[0](x3)
            x1, x2, x3 = self.net1.stages[1](x1), self.net2.stages[1](x2), self.net3.stages[1](x3)
            x1, x2, x3 = self.net1.stages[2](x1), self.net2.stages[2](x2), self.net3.stages[2](x3)
            x1, x2, x3 = tr_interaction(x1, x2, x3)
            x1, x2, x3 = self.net1.stages[3](x1), self.net2.stages[3](x2), self.net3.stages[3](x3)
            x1, x2, x3 = tr_interaction(x1, x2, x3)
            x1, x2, x3 = self.net1.stages[4](x1), self.net2.stages[4](x2), self.net3.stages[4](x3)
            x_cat = [x1,
                     upsample(x2, (int(x1.shape[-2]), int(x1.shape[-1]))),
                     upsample(x3, (int(x1.shape[-2]), int(x1.shape[-1])))]
            x_cat = torch.cat(x_cat, dim=1)
            return [x_cat]  # Return the concatenated feature
        else:
            raise NotImplementedError
