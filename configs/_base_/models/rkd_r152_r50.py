model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='RKD',
        low_feat_size=(4, 4),
        high_feat_size=(7, 7),
        pretrained_t='./checkpoints/tsn_r152_1x1x8_50e_actnet_rgb.pth',
        low_res_cfg=dict(
            type='ResNet',
            pretrained='torchvision://resnet50',
            depth=50,
            norm_eval=False),
        high_res_cfg=dict(
            type='ResNet',
            pretrained='torchvision://resnet152',
            depth=152,
            norm_eval=False)),
    cls_head=dict(
        type='RKD2DHead',
        num_classes=200,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        w_loss_kd=100.0,
        dropout_ratio=0.5,
        init_std=0.001),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))