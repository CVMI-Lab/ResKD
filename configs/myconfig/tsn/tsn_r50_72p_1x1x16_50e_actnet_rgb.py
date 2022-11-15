_base_ = [
    '../../_base_/models/tsn_r50.py', '../../_base_/schedules/sgd_50e.py',
    '../../_base_/default_runtime.py'
]
model = dict(cls_head=dict(num_classes=200))
# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/actnet/imgs'
data_root_val = 'data/actnet/imgs'
ann_file_train = 'data/actnet/actnet_png_img_train_split.txt'
ann_file_val = 'data/actnet/actnet_png_img_val_split.txt'
ann_file_test = 'data/actnet/actnet_png_img_val_split.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 80)),
    dict(
        type='MultiScaleCrop',
        input_size=80,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(72, 72), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 80)),
    dict(type='CenterCrop', crop_size=72),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 80)),
    dict(type='CenterCrop', crop_size=72),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        filename_tmpl='img_{:05}.png',
        multi_class=True,
        num_classes=200),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        filename_tmpl='img_{:05}.png',
        multi_class=True,
        num_classes=200),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        filename_tmpl='img_{:05}.png',
        multi_class=True,
        num_classes=200))
evaluation = dict(
    interval=5, metrics=['mean_average_precision'])

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/tsn_r50_72p_1x1x16_50e_actnet_rgb/'

