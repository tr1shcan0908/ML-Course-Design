"""Convert two pinned official models into audited tensor-only checkpoints."""
import hashlib
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import timm
import torch
from fer.emotieff import CLASS_ROWS, SOURCE_CLASSES, TARGET_CLASSES, VARIANTS, map_classifier
from fer.io import atomic_json
from fer.train import atomic_checkpoint


ALLOWED = {
    'timm.layers.adaptive_avgmax_pool.SelectAdaptivePool2d',
    'timm.layers.conv2d_same.Conv2dSame', 'timm.layers.norm_act.BatchNormAct2d',
    'timm.models._efficientnet_blocks.DepthwiseSeparableConv',
    'timm.models._efficientnet_blocks.InvertedResidual',
    'timm.models._efficientnet_blocks.SqueezeExcite',
    'timm.models.efficientnet.EfficientNet',
    'torch.nn.modules.activation.SiLU', 'torch.nn.modules.activation.Sigmoid',
    'torch.nn.modules.container.Sequential', 'torch.nn.modules.conv.Conv2d',
    'torch.nn.modules.flatten.Flatten', 'torch.nn.modules.linear.Identity',
    'torch.nn.modules.linear.Linear', 'torch.nn.modules.pooling.AdaptiveAvgPool2d',
    'torch.nn.modules.pooling.AdaptiveMaxPool2d',
}
SOURCES = {
    'b0': ('enet_b0_8_best_vgaf.pt', '95aafb39b8bb87964f45e208b9ab31e276e3e5278678db4961d18e6a1b42a141'),
    'b2': ('enet_b2_7.pt', 'e687fd9e7bee45486c7325f11222740f841c3bebac531badb903a50e82b9ac0f'),
}


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    directory = ROOT / 'artifacts/weights'
    for variant, (filename, expected) in SOURCES.items():
        source = directory / filename
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Official checkpoint hash changed: {filename}')
        names = torch.serialization.get_unsafe_globals_in_checkpoint(source)
        if not set(names) <= ALLOWED:
            raise RuntimeError(f'Unreviewed serialized types: {set(names) - ALLOWED}')
        objects = [getattr(importlib.import_module(n.rsplit('.', 1)[0]), n.rsplit('.', 1)[1]) for n in names]
        with torch.serialization.safe_globals(objects):
            original = torch.load(source, map_location='cpu', weights_only=True).eval()
        # timm 1.0 added optional space-to-depth and antialias modules. These
        # old checkpoints have neither operation; restore only their no-op defaults.
        compatibility = []
        for name, module in original.named_modules():
            if type(module).__name__ in ('DepthwiseSeparableConv','InvertedResidual'):
                for attribute in ('conv_s2d','bn_s2d'):
                    if not hasattr(module,attribute):
                        setattr(module,attribute,None)
                        compatibility.append(name+'.'+attribute+'=None')
                if not hasattr(module,'aa'):
                    module.aa = torch.nn.Identity()
                    compatibility.append(name+'.aa=Identity')
        spec = VARIANTS[variant]
        state = map_classifier(original.state_dict(), spec['source_classes'])
        recreated = timm.create_model(spec['architecture'], pretrained=False, num_classes=6,
                                      global_pool=spec['pool'], drop_rate=0., drop_path_rate=0.).eval()
        recreated.load_state_dict(state, strict=True)
        with torch.inference_mode():
            inputs = torch.randn(2, 3, spec['image_size'], spec['image_size'])
            expected_logits = original(inputs)[:, CLASS_ROWS[spec['source_classes']]]
            actual = recreated(inputs)
            torch.testing.assert_close(actual, expected_logits, rtol=2e-5, atol=2e-5)
        metadata = dict(variant=variant, **spec, source_file=filename, source_sha256=expected,
                        source_url=f'https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/520a051c64cd191521e5934655314e769a319684/models/affectnet_emotions/{filename}',
                        classes=TARGET_CLASSES, original_classes=SOURCE_CLASSES[spec['source_classes']],
                        classifier_rows=CLASS_ROWS[spec['source_classes']], timm=timm.__version__,
                        torch=str(torch.__version__), max_logit_error=float((actual-expected_logits).abs().max()),
                        compatibility_noops=compatibility,
                        verification='strict state load and full-resolution FP32 logit equivalence')
        output = directory / f'emotieff_{variant}_six_classes.pt'
        atomic_checkpoint(output, dict(model=state, metadata=metadata))
        metadata['converted_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
        atomic_json(output.with_suffix('.json'), metadata)
        print(metadata, flush=True)


if __name__ == '__main__':
    main()
