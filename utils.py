import math

import numpy as np


def calc_psnr(compressed, hr):
    """
    psnr是“Peak Signal to Noise Ratio”的缩写，即峰值信噪比，是一种评价图像的客观标准
    """
    diff = compressed - hr
    shave = 7
    valid = diff[shave:-shave, shave:-shave, ...]
    valid = valid.flatten('C')
    mse = np.mean(np.power(valid, 2))
    rmse = math.sqrt(np.mean(mse))
    if rmse < 1e-5:
        return 100
    return 20 * np.log10(255.0/rmse)
