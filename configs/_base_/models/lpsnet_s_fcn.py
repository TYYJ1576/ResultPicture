
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255
    )

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='LPSNet',
        depth=(1, 3, 3, 10, 10),
        width=(8, 24, 48, 96, 96),
        resolution=(3 / 4, 1 / 4, 0),
        deploy=False,
        init_cfg=None
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=96 * 2,  # Since we're concatenating features from xh and xl
        in_index=0,  # Adjust based on what your backbone returns
        channels=512,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
        )
    ),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
