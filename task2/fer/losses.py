"""NLA loss adaptation, preserving the authors' weighting and consistency formula.

Reference: third_party/nla/loss.py and train.py. This is a loss transfer onto
EmoNeXt, not a reproduction of the original ResNet18/MS-Celeb-1M experiment.
"""
import math

import torch
from torch.nn import functional as F


def eac_consistency(attention, flipped_attention):
    """Exact horizontal alignment, equivalent to EAC's integer flip grid."""
    if attention.shape != flipped_attention.shape or attention.ndim != 4:
        raise ValueError('EAC requires matching N,C,H,W class activation maps.')
    return F.mse_loss(attention.float(), flipped_attention.float().flip(-1))


def gaussian_weight(points, mean, diagonal, off_diagonal):
    covariance = points.new_tensor([[diagonal, off_diagonal], [off_diagonal, diagonal]])
    difference = points - points.new_tensor(mean)[:,None]
    exponent = -.5 * (difference * (torch.linalg.inv(covariance) @ difference)).sum(0)
    return torch.exp(exponent) / (2*math.pi*torch.sqrt(torch.det(covariance)))


def nla_loss(logits, flipped_logits, labels, epoch_index, epochs, reference_batch=32):
    # Compute covariance and divergences in FP32 under mixed precision.
    logits, flipped_logits = logits.float(), flipped_logits.float()
    probabilities = logits.softmax(1)
    ground_truth = probabilities.gather(1, labels[:,None]).squeeze(1)
    top_two = probabilities.topk(2, dim=1).values
    correct = top_two[:,0].eq(ground_truth)
    nearest_negative = torch.where(correct, top_two[:,1], top_two[:,0])
    points = torch.stack([ground_truth, nearest_negative])
    schedule = 1-math.exp(-15*epoch_index/max(epochs,1))
    true_major = .8**2 + (.8/2)**2
    true_minor = (-.8**2 + (.8/2)**2)*schedule
    false_major = .91**2 + (.91/6)**2
    false_minor = .91**2 - (.91/6)**2
    true_weight = gaussian_weight(points, [.5,.5], true_major, true_minor)
    false_weight = gaussian_weight(points, [.30,.15], false_major, false_minor)
    weights = torch.where(correct, true_weight, false_weight)
    ce = F.cross_entropy(logits, labels, reduction='none')
    # The public implementation uses a logits-mean target and reverse KL.
    # Preserve that behavior; do not silently substitute the standard JS formula.
    middle = ((logits+flipped_logits)*.5).softmax(1)
    consistency = .5*(F.kl_div(logits.log_softmax(1), middle, reduction='none').sum(1)
                       + F.kl_div(flipped_logits.log_softmax(1), middle, reduction='none').sum(1))
    # The original loss sums over its batch of 32. Make this scale explicit
    # so gradient accumulation does not alter its relative contribution.
    return .5*ce.mean()+.5*(ce*weights).mean()+.5*reference_batch*consistency.mean()
