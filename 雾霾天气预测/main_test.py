# encoding: utf-8
"""
@author: lee
@desc:
"""
from fun.read_data import read_data
from fun.sql_convert import sql_convert
from fun.normalized import normalized
from fun.split import split
from fun.logistic_regression import logistic_regression
from fun.prediction import prediction
from fun.random_forest import random_forest
from fun.two_category_assessment import two_category_assessment


def main_test():
    """
    :return:
    """
    # 读数据库表
    read_data(input0="./data/smog", output0="./data/read_data_out")
    # SQL脚本
    sql_convert(input0="./data/read_data_out", output0="./data/sql_convert_out")
    # 归一化
    normalized(field=[3, 4, 5, 6], input0="./data/sql_convert_out", output0="./data/normalized_out")
    # 拆分
    split(split_ratio=0.8, random_seed_num=1, input0="./data/normalized_out",
          output0="./data/split_out", output1="./data/split_out1")
    # 逻辑回归二分类
    logistic_regression(train_feature_column=[3, 4, 5, 6], target_column=2,
                        input0="./data/split_out", output0="./data/lr_out")
    # 预测
    prediction(input1="./data/lr_out", input0="./data/split_out1", feature_column=[3, 4, 5, 6],
               output0="./data/prediction_out")
    # 二分类评估
    two_category_assessment(original_label_column=2,
                            input0="./data/prediction_out", output0="./data/tca_out")
    # 随机森林
    random_forest(input0="./data/split_out", feature_column=[3, 4, 5, 6],
                  label_column=2, output0="./data/rf_out")
    # 预测
    prediction(input1="./data/rf_out", input0="./data/split_out1", feature_column=[3, 4, 5, 6],
               output0="./data/prediction_out1")
main_test()
