# coding=utf-8
from fun.read_data import *
from fun.missing_value_padding import *
from fun.linear_regression import *
from fun.prediction import *
from fun.random_sampling import *
from common.properties_util import *


def main():
    # 读数据表
    exec('read_data(' + Properties("./pro/read_data_input.properties").getProperties() + Properties(
        "./pro/read_data_out.properties").getProperties() + ' )')

    # 缺失值填充
    exec('missing_value_padding(' + Properties(
        "./pro/missing_value_padding_input.properties").getProperties() + Properties(
        "./pro/missing_value_padding_output.properties").getProperties() + ' )')

    # 随机采样
    exec('random_sampling(' + Properties(
        "./pro/random_sampling_input1.properties").getProperties() + Properties(
        "./pro/random_sampling_output1.properties").getProperties() + ' )')
    # 随机采样
    exec('random_sampling(' + Properties(
        "./pro/random_sampling_input2.properties").getProperties() + Properties(
        "./pro/random_sampling_output2.properties").getProperties() + ' )')
    # 线性回归
    exec('linear_regression(' + Properties(
        "./pro/linear_regression_input.properties").getProperties() + Properties(
        "./pro/linear_regression_output.properties").getProperties() + ' )')
    # 预测
    exec('prediction(' + Properties(
        "./pro/prediction_input1.properties").getProperties() + Properties(
        "./pro/prediction_input2.properties").getProperties() + Properties(
        "./pro/prediction_output.properties").getProperties() + ' )')
    # prediction(input="./data/rd_out2.csv",model="./data/mvd_out.pkl",feature_column=[0,2])


main()
