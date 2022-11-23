from typing import Union
import torch
import numpy as np
import math


__all__ = ['cbf']


def cbf(dM: Union[torch.Tensor, np.ndarray], M0: Union[torch.Tensor, np.ndarray],
        eps=1e-6, maxPerfusion=150, TR=4000, TE=9.1, PLD=1500, LabelTime=1800, NR=46,
        alpha=0.85, delta=2000, lamda=0.9, T1a=1624, T2a=80, T1RF=750):
    """
    Computing CBF

    :param dM:
    :param M0:
    :param maxPerfusion:
    :param TR: TR in msec
    :param TE: TE in msec
    :param PLD: Post-Labeling Delay (ms)
    :param LabelTime: Labeling duration (ms)
    :param NR: number of measurement
    :param alpha: labeling efficiency
    :param delta: arterial transit time (ms)
    :param lamda: blood-tissue water partition (mL/g)
    :param T1a: blood T1 (1624ms for human@3T; 2210ms rat@7T, 2430ms @9.4T based on Dobre et al, MRI 2007)
    :param T2a: blood T2-star (80ms for human@3T based on Wu WC, Neuroimage 2011)
    :param T1RF: (ms) based on Aslan, MRM 2010

    :return: CBF image
    """

    if isinstance(dM, np.ndarray):
        SLtime = np.arange(0, dM.shape[1])
        module = np
    else:
        SLtime = torch.arange(0, dM.shape[1], dtype=dM.dtype, device=dM.device)
        module = torch

    zeros = module.zeros_like(dM)
    # maxs = module.ones_like(dM) * maxPerfusion

    dM = module.where(M0 <= 0, zeros, dM)

    SLtime = SLtime.reshape((1, dM.shape[1], *tuple([1] * (dM.ndim - 2))))
    t = module.exp(-(PLD + SLtime) / T1a) - module.exp(-(PLD + SLtime + LabelTime) / T1a)

    dM = dM / 2 / alpha
    CBF = dM / (M0 + eps) / T1a / t * math.exp(TE / T2a)

    CBF = CBF * 60 * 100 * 1000 * lamda

    CBF = module.where(CBF < 0, zeros, CBF)
    CBF = module.where(CBF > maxPerfusion, zeros, CBF)  # maxs
    CBF = module.where(module.isnan(CBF), zeros, CBF)
    CBF = module.where(module.isinf(CBF), zeros, CBF)  # maxs

    return CBF
