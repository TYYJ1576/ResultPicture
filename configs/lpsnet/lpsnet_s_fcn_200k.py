# lpsnet_s_fcn.py

_base_ = [
    '../_base_/models/lpsnet_s_fcn.py',
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_200k.py'
]

crop_size = (768, 1536)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
