import cv2
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


def get_kuwahara_filtered_pic(full_filename: str, r: int = 10) -> ndarray:
    """
    return kuwahara filtered pic data as numpy ndarray
    """
    pic = np.array(plt.imread(full_filename))
    h, w, _ = pic.shape

    pic = np.pad(pic, ((r, r), (r, r), (0, 0)), "edge")
    ave, var = cv2.integral2(pic)
    ave = (ave[:-r - 1, :-r - 1] + ave[r + 1:, r + 1:] - ave[r + 1:, :-r - 1] - ave[:-r - 1, r + 1:]) / (
            r + 1) ** 2  # 平均値の一括計算
    var = ((var[:-r - 1, :-r - 1] + var[r + 1:, r + 1:] - var[r + 1:, :-r - 1] - var[:-r - 1, r + 1:]) / (
            r + 1) ** 2 - ave ** 2).sum(axis=2)  # 分散の一括計算

    def filter_kuwahara(i, j):
        return np.array([ave[i, j], ave[i + r, j], ave[i, j + r], ave[i + r, j + r]])[(
            np.array([var[i, j], var[i + r, j], var[i, j + r], var[i + r, j + r]]).argmin(axis=0).flatten(),
            j.flatten(),
            i.flatten())].reshape(w, h, _).transpose(1, 0, 2)

    return filter_kuwahara(*np.meshgrid(np.arange(h), np.arange(w))).astype(pic.dtype)  # 色の決定
