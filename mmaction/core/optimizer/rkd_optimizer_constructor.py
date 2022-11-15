# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmcv.utils import SyncBatchNorm, _BatchNorm, _ConvNd


@OPTIMIZER_BUILDERS.register_module()
class RKDOptimizerConstructor(DefaultOptimizerConstructor):
    """Optimizer constructor in TSM model.

    This constructor builds optimizer in different ways from the default one.

    1. Parameters of the first conv layer have default lr and weight decay.
    2. Parameters of BN layers have default lr and zero weight decay.
    3. If the field "fc_lr5" in paramwise_cfg is set to True, the parameters
       of the last fc layer in cls_head have 5x lr multiplier and 10x weight
       decay multiplier.
    4. Weights of other layers have default lr and weight decay, and biases
       have a 2x lr multiplier and zero weight decay.
    """

    def add_params(self, params, model):
        """Add parameters and their corresponding lr and wd to the params.

        Args:
            params (list): The list to be modified, containing all parameter
                groups and their corresponding lr and wd configurations.
            model (nn.Module): The model to be trained with the optimizer.
        """
        # use fc_lr5 to determine whether to specify higher multi-factor
        # for fc layer weights and bias.
        low_res_weight = []
        low_res_bias = []
        bn = []

        teacher_param = 'backbone.high_res_path'
        for n, m in model.named_modules():
            if n.startswith(teacher_param):
                continue
            else:
                if isinstance(m, _ConvNd):
                    m_params = list(m.parameters())
                    low_res_weight.append(m_params[0])
                    if len(m_params) == 2:
                        low_res_bias.append(m_params[1])
                elif isinstance(m, torch.nn.Linear):
                    m_params = list(m.parameters())
                    low_res_weight.append(m_params[0])
                    if len(m_params) == 2:
                        low_res_bias.append(m_params[1])
                elif isinstance(m, (_BatchNorm, SyncBatchNorm, torch.nn.GroupNorm)):
                    for param in list(m.parameters()):
                        if param.requires_grad:
                            bn.append(param)
                elif len(m._modules) == 0:
                    if len(list(m.parameters())) > 0:
                        raise ValueError(f'New atomic module type: {type(m)}. '
                                         'Need to give it a learning policy')

        # pop the cls_head fc layer params
        params.append({
            'params': low_res_weight,
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })
        params.append({
            'params': low_res_bias,
            'lr': self.base_lr * 2,
            'weight_decay': 0
        })
        params.append({'params': bn, 'lr': self.base_lr, 'weight_decay': 0})

