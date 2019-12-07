import copy as c


def deepcopy(cls, no_deep=[]):
    memo = dict()
    for key in no_deep:
        if hasattr(cls, key):
            memo[key] = getattr(cls, key)
            setattr(cls, key, None)
    cls_new = c.deepcopy(cls)
    for key, value in memo.items():
        setattr(cls, key, value)
        setattr(cls_new, key, value)
    return cls_new
