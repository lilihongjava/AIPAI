# encoding: utf-8
"""
@author: lee
@time: 2019/4/19 8:57
@file: LSTM.py
@desc: 
"""
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def lstm(step, input0, output0):
    data = numpy.genfromtxt(input0)
    data2 = data[:, 3]
    train_x, train_y = create_dataset(data2, step)
    # 一维要处理
    if train_y.ndim == 1:
        train_y = train_y[:, numpy.newaxis]
    # reshape input to be [samples, time steps, features]
    train_x = numpy.reshape(train_x, (train_x.shape[0], step, 1))
    # create and fit the LSTM network
    model = Sequential()
    print("~~~", train_x.shape)
    print("~~~", train_y.shape)
    model.add(LSTM(32, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_x, train_y, epochs=500, batch_size=1, verbose=2)
    scores = model.evaluate(train_x, train_y, verbose=0)
    print("Model Accuracy: %.2f%%", scores)
    # make predictions
    train_predict = model.predict(train_x)
    # print(train_predict)
    model.save(output0)


# convert an array of values into a dataset matrix
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
