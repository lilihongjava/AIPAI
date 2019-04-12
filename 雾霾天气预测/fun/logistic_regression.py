# encoding: utf-8
"""
@author: lee
@desc:
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.externals import joblib


def logistic_regression(train_feature_column, target_column, input0, output0):
    """
    :param train_feature_column:
    :param target_column:
    :param input:
    :param output:
    :return:
    """
    data = np.genfromtxt(input0)
    # print(data)
    df = pd.DataFrame(data)
    x_data = np.array(df.iloc[:, train_feature_column])
    y_data = data[:, target_column]
    # print(x_data)
    # plot(x_data,y_data)
    # plt.show()
    logistic = linear_model.LogisticRegression()
    logistic.fit(x_data, y_data)
    joblib.dump(logistic, output0)


def plot(x_data, y_data):
    """
    :param x_data:
    :param y_data:
    :return:
    """
    x0, x1, x2 = []
    y0, y1, y2 = []
    # 切分不同类别的数据
    for i in range(len(x_data)):
        if y_data[i] == 0:
            x0.append(x_data[i, 0])
            y0.append(x_data[i, 1])
        else:
            x1.append(x_data[i, 0])
            y1.append(x_data[i, 1])

    # 画图
    scatter0 = plt.scatter(x0, y0, c='b', marker='o')
    scatter1 = plt.scatter(x1, y1, c='r', marker='x')
    # scatter2 = plt.scatter(x2, y2, c='y', marker='*')
    # 画图例
    plt.legend(handles=[scatter0, scatter1], labels=['label0', 'label1'], loc='best')
