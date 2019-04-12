# encoding: utf-8
"""
@author: lee
@desc:
"""
import numpy as np


# 特殊处理
def sql_convert(input0, output0):
    """

    :param input0:
    :param output0:
    :return:
    """
    data = np.genfromtxt(input0)
    temp = data[:, 2]
    temp_length = len(temp)
    for i in range(temp_length):
        if temp[i] > 100:
            temp[i] = 1
        else:
            temp[i] = 0
    # print(data)
    np.savetxt(output0, data, fmt="%s")
