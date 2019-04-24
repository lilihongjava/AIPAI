# encoding: utf-8
"""
@author: lee
@desc:
"""
from fun.read_data import read_data
from fun.normalized import normalized
from fun.split_by_order import split_by_order
from fun.lstm import lstm
from fun.prediction_keras import prediction_keras


def main_test():
    """
    :return:
    """
    # 读数据库表
    read_data(input0="./data/csman", output0="./data/read_data_out")
    # 归一化
    normalized(field=[2], retain_original_column=True, input0="./data/read_data_out", output0="./data/normalized_out")
    # 拆分
    split_by_order(split_ratio=0.95, input0="./data/normalized_out", output0="./data/split_out",
                   output1="./data/split_out1")
    # LSTM
    # lstm(step=5, input0="./data/split_out", output0="./data/lstm_out")
    # 预测
    prediction_keras(input0="./data/split_out1", input1="./data/lstm_out",
                     feature_column=[3, 4, 5, 6], output0="./data/pre_out")

main_test()
