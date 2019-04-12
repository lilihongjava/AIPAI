# encoding: utf-8
"""
@author: lee
@desc:
"""
import numpy as np
import pandas as pd


# 按比例拆分
def split(split_ratio, input0, output0, output1, random_seed_num=None):
    """
    :param split_ratio:
    :param input0:
    :param output0:
    :param output1:
    :param random_seed_num:
    :return:
    """
    data = np.genfromtxt(input0)
    data_frame = pd.DataFrame(data)
    # print("data:" + data)
    df1 = data_frame.sample(frac=split_ratio, random_state=random_seed_num)
    np.savetxt(output0, df1.values, fmt="%s")
    # print(df1.shape)
    data_frame = data_frame.append(df1)
    # 求差集
    np.savetxt(output1, data_frame.drop_duplicates(keep=False), fmt="%s")
