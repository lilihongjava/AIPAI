# encoding: utf-8
"""
@author: lee
@desc:
"""
import numpy as np
import pandas as pd

from keras.models import load_model
import numpy
import sklearn
from sklearn.preprocessing import MinMaxScaler


def prediction_keras(feature_column, input0, input1, output0):
    model = load_model(input1)
    data = np.genfromtxt(input0)
    print(anti_normalization(data[:, 3], data[:, 2]))
    scaler = MinMaxScaler(feature_range=(0, 1))

    test_x = data[:, 2]
    # 一维要处理
    if test_x.ndim == 1:
        test_x = test_x[:, numpy.newaxis]
    test_x, test_y = create_dataset(data[:, 3], 5)
    # test_x, test_y = create_dataset(test_x, 5)
    test_x = numpy.reshape(test_x, (test_x.shape[0], 5, 1))
    test_predict = model.predict(test_x)
    test_predict = anti_normalization(test_predict, data[:, 2])
    np.savetxt(output0, test_predict, fmt="%s")


def anti_normalization(x, raw_data):
    return [(float(i)*float(max(raw_data)-min(raw_data)))+np.min(raw_data) for i in x]


def normalization(x):
    return [(float(i)-np.min(x))/float(max(x)-min(x)) for i in x]


def create_dataset(data, step=1):
    # data = dataset[:, 2]
    print("============", data.shape)
    # 一维要处理
    if data.ndim == 1:
        data = data[:, numpy.newaxis]
    data_x, data_y = [], []
    for i in range(len(data)-step-1):
        # 取step行数据
        a = data[i:(i+step), 0]
        data_x.append(a)
        data_y.append(data[i + step, 0])
    return numpy.array(data_x), numpy.array(data_y)
