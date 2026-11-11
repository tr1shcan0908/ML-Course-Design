"""Validation-only component replacement scoring and explicit module freezing."""
import hashlib
from pathlib import Path

import numpy as np

from .io import read_json


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def freeze_modules(model, config):
    names = list(config.get('freeze_modules', []))
    for stage in range(config.get('freeze_stages', 0)):
        names.extend([f'downsample_layers.{stage}', f'stages.{stage}'])
    modules = {name: model.get_submodule(name) for name in dict.fromkeys(names)}
    for module in modules.values():
        module.requires_grad_(False)
        module.eval()
    return modules


class ReplacementReference:
    """Fixed other-component predictions; never participates in gradient training."""

    def __init__(self, spec, indices, labels, classes, manifest_sha256):
        self.spec = spec
        self.indices = np.asarray(indices)
        self.labels = np.asarray(labels)
        self.classes = list(classes)
        self.weight = float(spec['candidate_weight'])
        self.fixed = np.zeros((len(labels), len(classes)), dtype=np.float32)
        if file_sha256(spec['replaces_checkpoint']) != spec['replaces_checkpoint_sha256']:
            raise ValueError('Replacement parent checkpoint hash mismatch.')
        if not 0 < self.weight < 1:
            raise ValueError('Candidate weight must be between zero and one.')
        if not np.isclose(self.weight + sum(c['weight'] for c in spec['components']), 1.):
            raise ValueError('Replacement weights must sum to one.')
        paths = [str(Path(c['checkpoint']).resolve()) for c in spec['components']]
        if len(paths) != len(set(paths)):
            raise ValueError('Reference checkpoints must be distinct.')
        if str(Path(spec['replaces_checkpoint']).resolve()) in paths:
            raise ValueError('Replaced model must not appear among fixed components.')
        for component in spec['components']:
            if component['weight'] <= 0:
                raise ValueError('Reference weights must be positive.')
            for field in ('checkpoint', 'probabilities', 'metadata'):
                if file_sha256(component[field]) != component[field + '_sha256']:
                    raise ValueError(f'Reference {field} hash mismatch.')
            metadata = read_json(component['metadata'])
            if (metadata['checkpoint_sha256'] != component['checkpoint_sha256'] or
                    metadata['manifest_sha256'] != manifest_sha256 or
                    metadata['flip_tta'] or metadata['eval_precision'] != 'float32' or
                    metadata['eval_batch_size'] != 32):
                raise ValueError('Reference prediction provenance mismatch.')
            if spec['classes'] != self.classes or spec['manifest_sha256'] != manifest_sha256:
                raise ValueError('Reference class order or manifest mismatch.')
            with np.load(component['probabilities'], allow_pickle=False) as saved:
                self.check_alignment(saved['labels'], saved['indices'])
                probs = saved['probabilities']
                self.check_probabilities(probs)
                self.fixed += component['weight'] * probs

    def check_alignment(self, labels, indices):
        if not np.array_equal(indices, self.indices) or not np.array_equal(labels, self.labels):
            raise ValueError('Reference validation indices or labels are misaligned.')

    def check_probabilities(self, probs):
        if (probs.shape != self.fixed.shape or not np.isfinite(probs).all() or
                np.any(probs < 0) or not np.allclose(probs.sum(1), 1., atol=1e-6, rtol=0)):
            raise ValueError('Invalid replacement probabilities.')

    def combine(self, labels, probabilities, indices):
        self.check_alignment(labels, indices)
        self.check_probabilities(probabilities)
        return self.fixed + self.weight * probabilities
