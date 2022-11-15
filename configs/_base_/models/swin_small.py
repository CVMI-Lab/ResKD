_base_ = "swin_tiny.py"
model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2],
        pretrained='./checkpoints/swin_small_imagenet1k_pretrain.pth',
        use_checkpoint=True))
