# encoding: utf-8
"""
@author: lee
@desc: 按比例拆分不打乱顺序
"""
import numpy as np


def split_by_order(split_ratio, input0, output0, output1):
    dataset = np.genfromtxt(input0)
    train_size = int(len(dataset) * split_ratio)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    np.savetxt(output0, train, fmt="%s")
    np.savetxt(output1, test, fmt="%s")
