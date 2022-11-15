import warnings
import torch
import torch.nn as nn
from ..builder import BACKBONES, build_backbone


@BACKBONES.register_module()
class RKD(nn.Module):
    def __init__(self,
                 low_res_cfg,
                 high_res_cfg,
                 low_img_res=(112, 112),
                 low_feat_size=(4, 4),
                 high_feat_size=(7, 7),
                 pretrained_t=None):
        super().__init__()
        self.pretrained_t = pretrained_t
        self.low_img_res = low_img_res
        self.low_feat_size = low_feat_size
        self.high_feat_size = high_feat_size
        self.low_res_path = build_backbone(low_res_cfg)
        self.high_res_path = build_backbone(high_res_cfg)

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        self.low_res_path.init_weights()
        self.high_res_path.init_weights()
        if self.pretrained_t:
            state_dict_t = torch.load(self.pretrained_t)['state_dict']
            new_state_dict_t = dict()
            tmpl = 'backbone.'
            for k, v in state_dict_t.items():
                if k.startswith(tmpl):
                    k = k[len(tmpl):]
                    new_state_dict_t[k] = v
            self.high_res_path.load_state_dict(new_state_dict_t)

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_test(x)

    def forward_train(self, x):
        # prepare input for the student
        x_shape = x.shape
        if len(x_shape) == 5:
            x_low_res = x.view((-1,) + x_shape[2:])
            x_low_res = nn.functional.interpolate(x_low_res, mode='bilinear', size=self.low_img_res)
            x_low_res = x_low_res.view(x_shape[:2] + x_low_res.shape[1:])
        else:
            x_low_res = nn.functional.interpolate(x, mode='bilinear', size=self.low_img_res)

        x_low_res = self.low_res_path(x_low_res)

        with torch.no_grad():
            # we intentionally unfreeze bn here, as suggested in https://arxiv.org/abs/1904.01866
            # self.high_res_path_a.eval()
            x_high_res = self.high_res_path(x).detach()

        # use bilinear interpolation to up-sample low-resolution feature maps to align with high-resolution feature maps
        feat_shape = x_low_res.shape
        if len(feat_shape) == 5:
            rebuild_out = x_low_res.reshape((-1,) + feat_shape[2:])
            rebuild_out = nn.functional.interpolate(rebuild_out, size=self.high_feat_size, mode='bilinear')
            rebuild_out = rebuild_out.reshape(feat_shape[:2] + rebuild_out.shape[1:])
        else:
            rebuild_out = nn.functional.interpolate(x_low_res, size=self.high_feat_size, mode='bilinear')

        return (x_low_res, rebuild_out, x_high_res)

    def forward_test(self, x):
        # prepare input for the student
        x_shape = x.shape
        if len(x_shape) == 5:
            x_low_res = x.view((-1,) + x_shape[2:])
            x_low_res = nn.functional.interpolate(x_low_res, mode='bilinear', size=self.low_img_res)
            x_low_res = x_low_res.view(x_shape[:2] + x_low_res.shape[1:])
        else:
            x_low_res = nn.functional.interpolate(x, mode='bilinear', size=self.low_img_res)

        return self.low_res_path(x_low_res)
