model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='RKD',
        low_feat_size=(4, 4),
        high_feat_size=(7, 7),
        pretrained_t=None,
        low_res_cfg=dict(
            type='ResNet3dSlowOnly',
            depth=50,
            pretrained='torchvision://resnet50',
            lateral=False,
            conv1_kernel=(1, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        high_res_cfg=dict(
            type='ResNet3dSlowOnly',
            depth=50,
            pretrained='torchvision://resnet50',
            lateral=False,
            conv1_kernel=(1, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False)),
    cls_head=dict(
        type='RKD3DHead',
        in_channels=2048,
        num_classes=200,
        spatial_type='avg',
        dropout_ratio=0.5,
        w_loss_kd=100.0),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
