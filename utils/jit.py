import torch

import configs


def script(fn):
    if configs.env.base.jit.script:
        try:
            fn = torch.jit.script(fn)
            # print(fn.code)
        except Exception as e:
            print(e)
    return fn


def compile(fn):
    if configs.env.base.jit.compile and hasattr(torch, 'compile'):
        try:
            fn = torch.compile(fn)
            # print(fn.code)
        except Exception as e:
            print(e)
    return fn
