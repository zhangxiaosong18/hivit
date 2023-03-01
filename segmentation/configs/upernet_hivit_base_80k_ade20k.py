_base_ = [
    '_base_/models/upernet_swin.py', '_base_/datasets/ade20k_640x640.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_80k.py'
]
crop_size = (640, 640)

pretrained = '../pretrain/mae_finetune_hivit_base_1600e.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='HiViT',
        img_size=224,
        task_img_size=640,
        patch_size=16,
        embed_dim=512,
        depths=[2, 2, 20],
        num_heads=8,
        mlp_ratio=4.,
        rpe=True,
        drop_path_rate=0.1,
        with_fpn=True,
        out_indices=['H', 'M', 19, 19],
        use_checkpoint=False,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(
        in_channels=[128, 256, 512, 512],
        num_classes=150,
        channels=1024,
    ),
    auxiliary_head=dict(
        in_channels=512,
        num_classes=150
    ),
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(426, 426))
)


optimizer = dict(_delete_=True, type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='HiViTLayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=20, layer_decay_rate=0.75))


lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=750,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=4)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (640, 640)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2560, 640),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            dict(type='SETR_Resize', keep_ratio=True,
                 crop_size=crop_size, setr_multi_scale=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

runner = dict(type='IterBasedRunnerAmp')

# do not use mmdet version fp16
# We set samples_per_gpu to 1 and optimizer_config.update_interval to 2, the total update step keep 160k.
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
