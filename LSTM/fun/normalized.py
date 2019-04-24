import numpy as np
import copy
import pandas as pd


def normalized(field, retain_original_column=False, input0="", output0=""):
    # np.set_printoptions(suppress=True)
    raw_data = np.genfromtxt(input0)
    # print(df)
    data = copy.deepcopy(raw_data)
    for i in field:
        data[:, i] = normalization(data[:, i])
        raw_data = np.column_stack((raw_data, data[:, i]))
    if retain_original_column:
        np.savetxt(output0, raw_data, fmt="%s")
    else:
        np.savetxt(output0, data, fmt="%s")


def normalization(x):
    return [(float(i)-np.min(x))/float(max(x)-min(x)) for i in x]
