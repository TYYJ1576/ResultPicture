# configs/lpsnet/lpsnet_s.py

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # ImageNet mean values
    std=[58.395, 57.12, 57.375],     # ImageNet std values
    to_rgb=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=img_norm_cfg['mean'],
        std=img_norm_cfg['std'],
        bgr_to_rgb=img_norm_cfg['to_rgb'],
        pad_val=0,
        seg_pad_val=255,
        size=(512, 1024)  # Specify if you want to ensure consistent input size
    ),
    backbone=dict(
        type='LPSNet',
        depth=(1, 3, 3, 10, 10),
        width=(8, 24, 48, 96, 96),
        resolution=(3 / 4, 1 / 4, 0),
        deploy=False,
        init_cfg=None
    ),
    decode_head=dict(
        type='LPSNetHead',
        in_channels=192,  # Adjust according to your model's output channels
        num_classes=19,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
        )
    ),
    # Model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

