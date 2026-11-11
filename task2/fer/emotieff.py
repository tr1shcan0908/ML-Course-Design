"""EmotiEffLib transfer with explicit pooling and six-class head mapping.

Official source: sb-ai-lab/EmotiEffLib, commit
520a051c64cd191521e5934655314e769a319684. No upstream code is executed.
"""
import torch
from torch import nn
from torch.nn import functional as F
import timm


TARGET_CLASSES = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
SOURCE_CLASSES = {
    7: ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'],
    8: ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'],
}
CLASS_ROWS = {7: [0, 2, 3, 4, 5, 6], 8: [0, 3, 4, 5, 6, 7]}
VARIANTS = {
    'b0': dict(architecture='tf_efficientnet_b0', pool='avg', image_size=224, source_classes=8),
    'b2': dict(architecture='tf_efficientnet_b2', pool='max', image_size=260, source_classes=7),
}


def map_classifier(state, source_classes):
    """Keep matching source logits, rather than randomly replacing the head."""
    if source_classes not in CLASS_ROWS:
        raise ValueError('Only the documented AffectNet seven/eight-class mappings are supported.')
    result = dict(state)
    for suffix in ('weight', 'bias'):
        source = result.pop('classifier.0.' + suffix, None)
        if source is None:
            source = result['classifier.' + suffix]
        if source.shape[0] != source_classes:
            raise ValueError('Classifier shape disagrees with the source class order.')
        result['classifier.' + suffix] = source[CLASS_ROWS[source_classes]].clone()
    return result


class FEREmotiEff(nn.Module):
    def __init__(self, variant='b0', dropout=0.):
        super().__init__()
        spec = VARIANTS[variant]
        self.backbone = timm.create_model(spec['architecture'], pretrained=False,
                                          num_classes=6, global_pool=spec['pool'],
                                          drop_rate=dropout, drop_path_rate=0.)

    def forward(self, x, return_aux=False, return_attention=False, attention_only=False):
        features = self.backbone.forward_features(x)
        if return_attention or attention_only:
            # EAC's public implementation detaches the classifier weights and
            # compares all unnormalised class maps. Gradients reach both views.
            with torch.autocast(device_type=x.device.type, enabled=False):
                attention = F.conv2d(features.float(),
                                     self.backbone.classifier.weight.detach().float()[:, :, None, None])
            if attention_only:
                return attention
        logits = self.backbone.forward_head(features)
        if return_attention:
            return logits, attention
        return (logits, logits.new_zeros(())) if return_aux else logits
