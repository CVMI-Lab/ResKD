# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet152',
        depth=152,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=400,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,),
    # model training and testing settings
    train_cfg=dict(average_clips='prob'),
    test_cfg=dict(average_clips='prob'))
