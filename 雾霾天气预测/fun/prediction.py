# encoding: utf-8
"""
@author: lee
@desc:
"""
import numpy as np
import pandas as pd
from sklearn.externals import joblib


def prediction(feature_column, input0, input1, output0):
    """
    :param feature_column:
    :param input0:
    :param input1:
    :param output0:
    :return:
    """
    model = joblib.load(input1)
    # 测试
    rawdata = np.genfromtxt(input0)
    # 一维要处理
    if rawdata.ndim == 1:
        data = rawdata[np.newaxis, :]
    data_frame = pd.DataFrame(rawdata)
    data = np.array(data_frame.iloc[:, feature_column])
    predict = model.predict(data)
    # print("predict:", predict)
    # print("true_labels:", rawdata[:, 2])
    scores = model.predict_proba(data)[::, 1]

    out = np.column_stack((rawdata, predict[:, np.newaxis]))
    out = np.column_stack((out, scores[:, np.newaxis]))
    np.savetxt(output0, out, fmt="%s")
