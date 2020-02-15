from typing import Union
import numpy as np
import torch
import torch.nn.functional as F


def sobel3d(data: Union[torch.Tensor, np.ndarray]):
    is_ndarray = False
    if isinstance(data, np.ndarray):
        is_ndarray = True
        data = torch.from_numpy(data)

    data_3d = torch.unsqueeze(data, 1)
    data_3d = F.pad(data_3d, (1, 1, 1, 1, 1, 1), 'replicate')

    f_x = torch.tensor(
        [
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            [[2, 0, -2], [8, 0, -8], [2, 0, -2]],
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        ], dtype=data.dtype, device=data.device).unsqueeze(0).unsqueeze(1)
    g_x = F.conv3d(data_3d, f_x)

    f_y = torch.tensor(
        [
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            [[2, 8, 2], [0, 0, 0], [-2, -8, -2]],
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        ], dtype=data.dtype, device=data.device).unsqueeze(0).unsqueeze(1)
    g_y = F.conv3d(data_3d, f_y)

    f_z = torch.tensor(
        [
            [[1, 2, 1], [2, 8, 2], [1, 2, 1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[-1, -2, -1], [-2, -8, -2], [-1, -2, -1]]
        ], dtype=data.dtype, device=data.device).unsqueeze(0).unsqueeze(1)
    g_z = F.conv3d(data_3d, f_z)

    sobel = torch.sqrt((g_x ** 2 + g_y ** 2 + g_z ** 2) / 3)
    sobel = sobel.reshape(sobel.size(0), *sobel.shape[2:])

    if is_ndarray:
        sobel = sobel.numpy()

    return sobel


def laplace3d(data: Union[torch.Tensor, np.ndarray]):
    is_ndarray = False
    if isinstance(data, np.ndarray):
        is_ndarray = True
        data = torch.from_numpy(data)

    data_3d = torch.unsqueeze(data, 1)
    data_3d = F.pad(data_3d, (1, 1, 1, 1, 1, 1), 'replicate')

    f = torch.tensor(
        [
            [[0, -1, 0], [-1, -1, -1], [0, -1, 0]],
            [[-1, -1, -1], [-1, 18, -1], [-1, -1, -1]],
            [[0, -1, 0], [-1, -1, -1], [0, -1, 0]]
        ], dtype=data.dtype, device=data.device).unsqueeze(0).unsqueeze(1)

    laplace = F.conv3d(data_3d, f)
    laplace = laplace.reshape(laplace.size(0), *laplace.shape[2:])

    if is_ndarray:
        laplace = laplace.numpy()

    return laplace
