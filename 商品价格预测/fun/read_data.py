# encoding: utf-8
"""
@author: lee
@desc:
"""
import numpy as np


def read_data(input0, output0):
    # 载入数据
    data = np.genfromtxt(input0, delimiter=",")
    # print(data)
    np.savetxt(output0, data)
