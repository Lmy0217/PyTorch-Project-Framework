import copy
import functools
import os
import random

import numpy as np
import torch

__all__ = [
    'set_seed',
    'deepcopy', 'merge_dict', 'hasattrs',
    'cmp_class', 'is_abstract',
    'all_subclasses', 'all_subclasses_not_abstract'
]


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def deepcopy(cls, no_deep=()):
    memo = {}
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
            value = value.unsqueeze(-1)
            if key in dst.keys():
                dst[key] = torch.cat([dst[key], value.detach()])
            else:
                dst[key] = value.detach()
        else:
            if key in dst.keys():
                dst[key].append(value)
            else:
                dst[key] = [value]


def _all_subclasses(cls):
    return list(set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in _all_subclasses(c)]))


def cmp_class(cls1: type, cls2: type):
    if cls1.__name__ < cls2.__name__:
        return -1
    if cls1.__name__ > cls2.__name__:
        return 1
    return 0


def all_subclasses(cls):
    all_sub = _all_subclasses(cls)
    return sorted(all_sub, key=functools.cmp_to_key(cmp_class))


def is_abstract(cls):
    return bool(getattr(cls, '__abstractmethods__', False))


def all_subclasses_not_abstract(cls):
    return [c for c in all_subclasses(cls) if not is_abstract(c)]


def hasattrs(cls, attrs: list):
    flag = True
    for attr in attrs:
        flag = flag and hasattr(cls, attr)
    return flag
