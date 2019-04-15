# encoding: utf-8
"""
@author: lee
@desc:
"""
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.externals import joblib


def linear_regression(input, feature_column, label_column, output):
    """
    :param input:
    :param feature_column:
    :param label_column:
    :param output:
    :return:
    """
    data = np.genfromtxt(input)
    # print(data)
    data_frame = pd.DataFrame(data)
    # print(df.iloc[:, [0, 2]])
    x_data = np.array(data_frame.iloc[:, feature_column])
    y_data = data[:, label_column]
    # print(x_data)
    # x_data = x_data[:,np.newaxis]
    # y_data = y_data[:,np.newaxis]
    # print(x_data)
    #  创建并拟合模型
    model = linear_model.LinearRegression()
    model.fit(x_data, y_data)
    # print(model)
    # print(y_data)
    joblib.dump(model, output)
