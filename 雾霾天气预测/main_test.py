from fun.read_data import read_data
from fun.sql_conver import sql_conver
from fun.normalized import normalized
from fun.split import split
from fun.logistic_regression import logistic_regression

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
    logistic_regression(input="./data/split_out", output="./data/lr_out")
main_test()
