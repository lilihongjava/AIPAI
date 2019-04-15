# encoding: utf-8
"""
@author: lee
@desc:
"""
from fun.read_data import read_data
from fun.missing_value_padding import missing_value_padding
from fun.linear_regression import linear_regression
from fun.prediction import prediction
from fun.random_sampling import random_sampling


def main_test():
    # 读数据表
    read_data(input0="./data/job.csv", output0="./data/read_data_out.csv")
    # 缺失值填充
    missing_value_padding(input="./data/read_data_out.csv", filled_field="1,2,3", replace_with="2",
                          output="./data/mvd_out.csv")
    # 随机采样
    random_sampling(input="./data/mvd_out.csv", num_of_samples=7, sampling_ratio=0.15,
                    replace=True, random_seed_num=10, output="./data/rd_out1.csv")
    # 随机采样
    random_sampling(input="./data/mvd_out.csv", num_of_samples=1, sampling_ratio=0.15,
                    replace=True, random_seed_num=10, output="./data/rd_out2.csv")
    # 线性回归
    linear_regression(input="./data/rd_out1.csv", feature_column=[0, 2], label_column=1,
                      output="./data/mvd_out.pkl")
    # 预测
    prediction(input="./data/rd_out2.csv", model="./data/mvd_out.pkl", feature_column=[0, 2],
               output="./data/prediction_out.csv")
main_test()
