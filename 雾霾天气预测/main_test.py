from fun.read_data import read_data
from fun.sql_conver import sql_conver
from fun.normalized import normalized
from fun.split import split
from fun.logistic_regression import logistic_regression
from fun.prediction import prediction
from fun.random_forest import random_forest

def main_test():
    #读数据库表
    read_data(input="./data/wumai", output="./data/read_data_out")
    #SQL脚本
    sql_conver(input="./data/read_data_out", output="./data/sql_conver_out")
    #归一化
    normalized(field =[3,4,5,6],input="./data/sql_conver_out", output="./data/normalized_out")
    #拆分
    split(split_ratio=0.8,random_seed_num=1234567,input="./data/normalized_out", output="./data/split_out", output1="./data/split_out1")
    #逻辑回归二分类
    logistic_regression(train_feature_column=[3,4,5,6],target_column=2,input="./data/split_out", output="./data/lr_out")
    #预测
    prediction(input1="./data/lr_out",input="./data/split_out1",feature_column=[3,4,5,6],output="./data/prediction_out")
    #随机森林
    random_forest(input="./data/split_out", feature_column=[3,4,5,6],label_column=2,output="./data/rf_out")
    # 预测
    prediction(input1="./data/rf_out", input="./data/split_out1", feature_column=[3, 4, 5, 6],output="./data/prediction_out1")
main_test()
