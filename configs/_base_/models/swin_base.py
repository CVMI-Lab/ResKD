_base_ = "swin_tiny.py"
model = dict(backbone=dict(depths=[2, 2, 18, 2],
                           embed_dim=128,
                           num_heads=[4, 8, 16, 32],
                           pretrained='./checkpoints/swin_base_imagenet22k_pretrain.pth'),
             cls_head=dict(in_channels=1024))
