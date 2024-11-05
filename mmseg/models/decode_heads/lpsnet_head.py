# mmseg/models/decode_heads/lpsnet_head.py

import torch
import torch.nn as nn
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

@MODELS.register_module()
class LPSNetHead(BaseDecodeHead):
    def __init__(self, in_channels, num_classes, init_cfg=None, **kwargs):
        super(LPSNetHead, self).__init__(in_channels=in_channels,
                                         channels=in_channels,
                                         num_classes=num_classes,
                                         input_transform=None,
                                         init_cfg=init_cfg,
                                         **kwargs)
        self.conv_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        """Forward function."""
        x = inputs[0]  # Get the feature from the backbone
        x = self.cls_seg(x)
        return x

