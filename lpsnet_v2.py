import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule

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
    o3 = x3 + upsample(x1, s3) + upsample(x2, s3)
    return o1, o2, o3

class ConvBNReLU3x3(ConvModule):
    def __init__(self, c_in, c_out, stride, deploy=False):
        super(ConvBNReLU3x3, self).__init__(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=deploy,
            conv_cfg=None,
            norm_cfg=None if deploy else dict(type='BN'),
            act_cfg=dict(type='ReLU', inplace=True)
        )

class BaseNet(nn.Module)
    def __init__(self, layers, channels, deploy=False):
        super(BaseNet, self).__init__()
        self.layer = layers
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

@BACKBONES.register_module()
class LPSNet1Path(BaseModule):
    def __init__(self, depth, width, resolution, deploy=False, init_cfg=None):
        super(LPSNet1Path, self).__init__(init_cfg)
        self.depth = depth
        self.width = width
        self.resolution = resolution
        self.net = BaseNet(self.depth, self.width, deploy)

    def forward(self, x):
        x = self._preprocess_input(x)
        outputs = self.net(x)
        return outputs

    def _preprocess_input(self, x):
        r = self.resolution[0]
        return upsample(x, (int(x.shape[-2] * r), int(x.shape[-1] * r)))

@BACKBONES.register_module()
class LPSNet2Path(BaseModule):
    def __init__(self, depth, width, resolution, deploy=False, init_cfg=None):
        super(LPSNet2Path, self).__init__(init_cfg)
        self.depth = depth
        self.width = width
        self.resolution = resolution
        self.netH = BaseNet(self.depth, self.width, deploy)
        self.netL = BaseNet(self.depth, self.width, deploy)

    def forward(self, x):
        xh, xl = self._preprocess_input(x)
        outputs = []
        # Process through stages with bi_interaction
        xh_stage_outputs = []
        xl_stage_outputs = []

        for i in range(len(self.netH.stages)):
            xh = self.netH.stages[i](xh)
            xl = self.netL.stages[i](xl)
            if i >= 2:  # Apply bi_interaction after certain stages
                xh, xl = bi_interaction(xh, xl)
            xh_stage_outputs.append(xh)
            xl_stage_outputs.append(xl)

        # Collect outputs (you may need to adjust which outputs to return)
        outputs = [
            torch.cat([xh_stage_outputs[-1], upsample(xl_stage_outputs[-1], xh_stage_outputs[-1].shape[-2:])], dim=1)
        ]
        return outputs

    def _preprocess_input(self, x):
        r1, r2 = self.resolution[0], self.resolution[1]
        x1 = upsample(x, (int(x.shape[-2] * r1), int(x.shape[-1] * r1)))
        x2 = upsample(x, (int(x.shape[-2] * r2), int(x.shape[-1] * r2)))
        return x1, x2

@BACKBONES.register_module()
class LPSNet3Path(BaseModule):
    def __init__(self, depth, width, resolution, deploy=False, init_cfg=None):
        super(LPSNet3Path, self).__init__(init_cfg)
        self.depth = depth
        self.width = width
        self.resolution = resolution
        self.netH = BaseNet(self.depth, self.width, deploy)
        self.netM = BaseNet(self.depth, self.width, deploy)
        self.netL = BaseNet(self.depth, self.width, deploy)

    def forward(self, x):
        xh, xm, xl = self._preprocess_input(x)
        outputs = []
        # Process through stages with bi_interaction
        xh_stage_outputs = []
        xm_stage_outputs = []
        xl_stage_outputs = []
        
        for i in range(len(self.netH.stages)):
            xh = self.netH.stages[i](xh)
            xm = self.netM.stages[i](xm)
            xl = self.netL.stages[i](xl)
            if i >= 2:  # Apply bi_interaction after certain stages
                xh, xm, xl = tr_interaction(xh, xm, xl)
            xh_stage_outputs.append(xh)
            xm_stage_outputs.append(xm)
            xl_stage_outputs.append(xl)

        outputs = [
            torch.cat([xh_stage_output[-1], upsample(xm_stage_outputs[-1], xh_stage_outputs[-1].shape[-2:]), upsample(xl_stage_outputs[-1], xh_stage_outputs[-1].shape[-2:])], dim=1)
        ]
        return outputs

    def _preprocess_input(self, x):
        r1, r2, r3 = self.resolution[0], self.resolution[1], self.resolution[2]
        x1 = upsample(x, (int(x.shape[-2] * r1), int(x.shape[-1] * r1)))
        x2 = upsample(x, (int(x.shape[-2] * r2), int(x.shape[-1] * r2)))
        x3 = upsample(x, (int(x.shape[-2] * r3), int(x.shape[-1] * r3)))
        return x1, x2, x3

def build_lpsnet(depth, width, resolution, deploy=False):
    resolution_filter = list(filter(lambda x: x > 0, resolution))
    resolution_sorted = sorted(resolution_filter, reverse=True)
    if len(resolution_sorted) == 1:
        return LPSNet1Path(depth, width, resolution_sorted, deploy=deploy)
    elif len(resolution_sorted) == 2:
        return LPSNet2Path(depth, width, resolution_sorted, deploy=deploy)
    elif len(resolution_sorted) == 3:
        return LPSNet3Path(depth, width, resolution_sorted, deploy=deploy)
    else:
        raise NotImplementedError

