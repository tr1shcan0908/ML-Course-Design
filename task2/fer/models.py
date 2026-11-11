from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from third_party.emonext.models import EmoNeXt


class FERConvNeXt(EmoNeXt):
    def __init__(self, variant='emonext', num_classes=6, backbone='tiny', drop_path_rate=.1):
        if backbone not in ('tiny', 'small'):
            raise ValueError(f'Unknown ConvNeXt backbone: {backbone}')
        super().__init__(num_classes=num_classes, drop_path_rate=drop_path_rate,
                         depths=[3,3,27,3] if backbone=='small' else [3,3,9,3])
        self.variant = variant
        if variant == 'convnext':
            self.localization = nn.Identity()
            self.fc_loc = nn.Identity()
            self.attention = nn.Identity()
            for layer in self.downsample_layers[1:]:
                layer[2] = nn.Identity()

    def stn(self, x):
        if self.variant == 'convnext':
            return x
        # flatten supports channels-last tensors; upstream view assumes contiguous NCHW.
        theta = self.fc_loc(self.localization(x).flatten(1)).reshape(-1,2,3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        return F.grid_sample(x, grid, align_corners=True)

    def forward(self, x, return_aux=False):
        features = self.forward_features(self.stn(x))
        logits = self.head(features)
        if not return_aux:
            return logits
        regularizer = logits.new_zeros(())
        if self.variant != 'convnext':
            _, weights = self.attention(features)
            regularizer = (weights - weights.mean()).square().mean()
        return logits, regularizer


def make_model(config, load_pretrained=True):
    if config['model'] == 'emotieff':
        from .emotieff import FEREmotiEff, TARGET_CLASSES, VARIANTS
        variant = config['backbone']
        if config.get('image_size',224) != VARIANTS[variant]['image_size']:
            raise ValueError('Use the official EmotiEff input resolution for this comparison.')
        model = FEREmotiEff(variant, dropout=config.get('dropout',0.))
        info = dict(pretraining='EmotiEffLib: face identification then AffectNet expression',
                    backbone=variant, classes=TARGET_CLASSES, pool=VARIANTS[variant]['pool'])
        if load_pretrained:
            checkpoint = torch.load(config['pretrained'],map_location='cpu',weights_only=True)
            if checkpoint['metadata']['classes'] != TARGET_CLASSES or checkpoint['metadata']['variant'] != variant:
                raise RuntimeError('Converted pretrained label mapping or architecture mismatch.')
            model.backbone.load_state_dict(checkpoint['model'],strict=True)
            info.update(checkpoint['metadata'])
        return model, info
    if config['model'] == 'qcs':
        from third_party.qcs.models.QCS_7cls_raf_db import pyramid_trans_expr
        model = pyramid_trans_expr(num_classes=6, pretrained_path=config['pretrained'] if load_pretrained else None)
        # Unused face-identification output layer is never called by upstream forward.
        model.ir_back.output_layer = nn.Identity()
        unmatched = [key for key in model.ir_back.state_dict()
                     if load_pretrained and key not in model.ir_back._pretrained_matched and not key.endswith('num_batches_tracked')]
        if unmatched:
            raise RuntimeError(f'IR50 backbone weights not fully initialized: {unmatched[:10]}')
        return model, {'pretraining':'MS-Celeb-1M IR50', 'classes':6,
                       'loaded_tensors':len(getattr(model.ir_back, '_pretrained_matched', []))}
    backbone = config.get('backbone','tiny')
    model = FERConvNeXt(variant=config['model'], num_classes=6, backbone=backbone,
                       drop_path_rate=config.get('drop_path_rate',.1))
    info = {'pretraining':f'ImageNet-22K ConvNeXt-{backbone.title()}', 'classes':6}
    if load_pretrained:
        checkpoint = torch.load(Path(config['pretrained']), map_location='cpu', weights_only=True)
        state = checkpoint['model']
        state = {k:v for k,v in state.items() if not k.startswith('head.')}
        result = model.load_state_dict(state, strict=False)
        expected = ('head.', 'localization.', 'fc_loc.', 'attention.')
        unexpected_missing = [k for k in result.missing_keys
                              if not k.startswith(expected) and '.2.fc.' not in k]
        if result.unexpected_keys or unexpected_missing:
            raise RuntimeError(f'Unexpected pretrained mismatch: {result.unexpected_keys}/{unexpected_missing}')
        info.update(loaded_tensors=len(state), missing_task_tensors=result.missing_keys)
    return model, info
