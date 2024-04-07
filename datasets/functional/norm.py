import numpy as np
import torch


__all__ = [
    'standard_norm', 'standard_renorm',
    'triple_norm', 'triple_renorm',
    'threshold_norm', 'threshold_renorm'
]


def standard_norm(data, mean, std):
    assert (std > 0).all()
    return (data - mean) / std


def standard_renorm(data, mean, std):
    return data * std + mean


def triple_norm(data, mean, std):
    return standard_norm(data, mean, 3 * std)


def triple_renorm(data, mean, std):
    return standard_renorm(data, mean, 3 * std)


def threshold_norm(data, lower, upper):
    assert lower < upper
    if isinstance(data, torch.Tensor):
        data = torch.clip(data, min=lower, max=upper)
    else:
        data = np.clip(data, a_min=lower, a_max=upper)
    data = (data - lower) / (upper - lower)
    data = (data - 0.5) / 0.5
    return data


def threshold_renorm(data, lower, upper):
    data = data * 0.5 + 0.5
    data = data * (upper - lower) + lower
    return data
