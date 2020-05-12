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


def all_subclasses(cls):
    return list(set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)]))


def is_abstract(cls):
    return bool(getattr(cls, '__abstractmethods__', False))


def all_subclasses_not_abstract(cls):
    return [c for c in all_subclasses(cls) if not is_abstract(c)]


def hasattrs(cls, attrs: list):
    flag = True
    for attr in attrs:
        flag = flag and hasattr(cls, attr)
    return flag
