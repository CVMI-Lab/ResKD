# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import torch.nn.functional as F

from ..builder import HEADS
from .base import BaseHead
from ...core import top_k_accuracy


@HEADS.register_module()
class GeneralRKD3DHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 w_loss_kd=1.0,
                 pretrained_t=None,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        self.kd_loss = nn.KLDivLoss(reduction='mean')
        self.w_loss_kd = w_loss_kd
        self.pretrained_t = pretrained_t

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        self.fc_cls_t = nn.Linear(1024, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

        state_dict_t = torch.load(self.pretrained_t)['state_dict']
        new_state_dict_t = dict()
        tmpl = 'cls_head.fc_cls.'
        for k, v in state_dict_t.items():
            if k.startswith(tmpl):
                k = k[len(tmpl):]
                new_state_dict_t[k] = v
        self.fc_cls_t.load_state_dict(new_state_dict_t)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score

    def loss(self, cls_score, labels, rebuild_x, ori_x, **kwargs):
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if not self.multi_class and cls_score.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(
                top_k_acc[1], device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, labels, **kwargs)

        T = 2
        ori_x = ori_x.permute((0, 2, 3, 4, 1))
        ori_x = ori_x.reshape(-1, ori_x.shape[-1])
        rebuild_x = rebuild_x.permute((0, 2, 3, 4, 1))
        rebuild_x = rebuild_x.reshape(-1, rebuild_x.shape[-1])
        logit_map = self.fc_cls(rebuild_x)
        with torch.no_grad():
            logit_map_t = self.fc_cls_t(ori_x)
        loss_kd = self.w_loss_kd * T * T * self.kd_loss(F.log_softmax(logit_map / T, dim=1),
                                                        F.softmax(logit_map_t.detach() / T, dim=1))

        # loss_cls may be dictionary or single tensor
        if isinstance(loss_kd, dict):
            # losses.update(loss_cls)
            losses.update(loss_kd)
        else:
            losses['loss_cls'] = loss_cls
            losses['loss_kd'] = loss_kd

        return losses

