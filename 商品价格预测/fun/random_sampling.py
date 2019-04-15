# encoding: utf-8
"""
@author: lee
@desc:
"""
import numpy as np
import pandas as pd


def random_sampling(input=None, num_of_samples=None, sampling_ratio=None,
                    replace=None, random_seed_num=None, output=None):
    """
    :param input:
    :param num_of_samples:
    :param sampling_ratio:
    :param replace:
    :param random_seed_num:
    :param output:
    :return:
    """
    data = np.genfromtxt(input)
    # print(data)
    data_frame = pd.DataFrame(data)
    if num_of_samples is not None:
        data_frame = data_frame.sample(n=num_of_samples,
                                       replace=replace, random_state=random_seed_num)
    elif sampling_ratio is not None:
        data_frame = data_frame.sample(frac=sampling_ratio,
                                       replace=replace, random_state=random_seed_num)
    np.savetxt(output, data_frame.values)
