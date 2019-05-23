# encoding: utf-8
"""
@author: lee
@time: 2019/5/22 9:43
@file: multiple_time_lstm.py
@desc: 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def main():
    columns = ['YEAR', 'MONTH', 'DAY', 'TEMP_HIG', 'TEMP_COL', 'AVG_TEMP', 'AVG_WET', 'DATA_COL']
    data = pd.read_csv('./data/industry_timeseries/timeseries_train_data/1.csv',
                          names=columns)
    print(data.head())

    # 查看数据采集区1的数据
    plt.figure(figsize=(24, 8))
    for i in range(8):
        plt.subplot(8, 1, i + 1)
        plt.plot(data.values[:, i])
        plt.title(columns[i], y=0.5, loc='right')
    plt.show()
    # 将数据归一化到0-1之间,无量纲化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['DATA_COL', 'TEMP_HIG', 'TEMP_COL', 'AVG_TEMP', 'AVG_WET']].values)
    # np.savetxt("1.csv", scaled_data, fmt="%s")
    # 将时序数据转换为监督问题数据
    reframed = series_to_supervised(scaled_data, 1, 1)
    # np.savetxt("2.csv", reframed, fmt="%s")
    # 删除无用的label数据
    reframed.drop(reframed.columns[[6, 7, 8, 9]], axis=1, inplace=True)
    # np.savetxt("3.csv", reframed, fmt="%s")
    # print(reframed.info())
    # print(reframed.head())

    # 数据集划分,选取前400天的数据作为训练集,中间150天作为验证集,其余的作为测试集
    train_days = 400
    valid_days = 150
    values = reframed.values
    train = values[:train_days, :]
    valid = values[train_days:train_days + valid_days, :]
    test = values[train_days + valid_days:, :]
    train_X, train_y = train[:, :-1], train[:, -1]
    valid_X, valid_y = valid[:, :-1], valid[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    print(train_X.shape, train_y.shape, valid_X.shape, valid_y.shape, test_X.shape, test_y.shape)
    # 将数据集重构为符合LSTM要求的数据格式,即 [样本，时间步，特征]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    valid_X = valid_X.reshape((valid_X.shape[0], 1, valid_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, valid_X.shape, valid_y.shape, test_X.shape, test_y.shape)
    model1 = Sequential()
    print(train_X.shape[1], train_X.shape[2])

    model1.add(LSTM(50, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
    model1.add(Dense(units=1, activation='linear'))
    model1.compile(loss='mean_squared_error', optimizer='adam')

    # fit network
    history = model1.fit(train_X, train_y, epochs=100, batch_size=32,
                             validation_data=(valid_X, valid_y), verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.legend()
    plt.show()

    plt.figure(figsize=(24, 8))
    train_predict = model1.predict(train_X)
    valid_predict = model1.predict(valid_X)
    test_predict = model1.predict(test_X)
    plt.plot(values[:, -1], c='b')
    plt.plot([x for x in train_predict], c='g')
    plt.plot([None for _ in train_predict] + [x for x in valid_predict], c='y')
    plt.plot([None for _ in train_predict] + [None for _ in valid_predict] + [x for x in test_predict], c='r')
    plt.show()


if __name__ == "__main__":
    main()
