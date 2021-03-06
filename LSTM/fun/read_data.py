# encoding: utf-8
"""
@author: lee
@desc:
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def read_data(input0, output0):
    """
    :param input0:
    :param output0:
    :return:
    """
    if os.path.isdir(input0):
        # 返回一个列表其中包含在目录条目的名称
        files = os.listdir(input0)
        # 载入数据
        data = np.genfromtxt(input0 + files[0], delimiter="|")
    elif os.path.isfile(input0):
        data = np.genfromtxt(input0, delimiter="|")
    # print(data[:, 2])
    np.savetxt(output0, data)
    # 画图第三列指标数据
    plt.plot(data[:, 2])
    # plt.show()
