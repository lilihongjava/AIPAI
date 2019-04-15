# encoding: utf-8
"""
@author: lee
@desc:
"""
import numpy as np
import pandas as pd
from sklearn.externals import joblib


def prediction(input, model, feature_column, output):
    """
    :param input:
    :param model:
    :param feature_column:
    :param output:
    :return:
    """
    model = joblib.load(model)
    # 测试
    data = np.genfromtxt(input)
    # 一维要处理
    if data.ndim == 1:
        data = data[np.newaxis, :]
    data_frame = pd.DataFrame(data)
    # print(df)
    data = np.array(data_frame.iloc[:, feature_column])
    # print(data)
    predict = model.predict(data)
    print("predict:", predict)
    np.savetxt(output, predict)
