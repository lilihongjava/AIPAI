# encoding: utf-8
"""
@author: lee
@desc:
"""
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.externals import joblib


def logistic_regression(train_feature_column, target_column, input0, output0):
    """
    :param train_feature_column:
    :param target_column:
    :param input0:
    :param output0:
    :return:
    """
    data = np.genfromtxt(input0)
    # print(data)
    data_frame = pd.DataFrame(data)
    x_data = np.array(data_frame.iloc[:, train_feature_column])
    y_data = data[:, target_column]
    # print(x_data)
    logistic = linear_model.LogisticRegression()
    logistic.fit(x_data, y_data)
    joblib.dump(logistic, output0)
