# encoding: utf-8
"""
@author: lee
@desc:
"""
import numpy as np


def normalized(field, input0, output0):
    """

    :param field:
    :param input0:
    :param output0:
    :return:
    """
    # np.set_printoptions(suppress=True)
    data = np.genfromtxt(input0)
    # df = pd.DataFrame(data)
    # print(df)
    for i in field:
        data[:, i] = normalization(data[:, i])
    np.savetxt(output0, data)


def normalization(data):
    """
    :param data:
    :return:
    """
    return [(float(i)-np.min(data))/float(max(data)-min(data)) for i in data]
