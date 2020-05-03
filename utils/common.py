import copy
import torch


def deepcopy(cls, no_deep=list()):
    memo = dict()
    for key in no_deep:
        if hasattr(cls, key):
            memo[key] = getattr(cls, key)
            setattr(cls, key, None)
    cls_new = copy.deepcopy(cls)
    for key, value in memo.items():
        setattr(cls, key, value)
        setattr(cls_new, key, value)
    return cls_new


def merge_dict(dst: dict, src: dict):
    for key, value in src.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        if key in dst.keys():
            dst[key].append(value)
        else:
            dst[key] = [value]
