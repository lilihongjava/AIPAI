import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib

def linear_regression(input,feature_column,label_column,output):
    data = np.genfromtxt(input)
    #print(data)
    df = pd.DataFrame(data)
    #print(df.iloc[:, [0, 2]])
    x_data = np.array(df.iloc[:, feature_column])
    y_data = data[:,label_column]
    #print(x_data)
    #x_data = x_data[:,np.newaxis]
    #y_data = y_data[:,np.newaxis]
    #print(x_data)
    #  创建并拟合模型
    model = linear_model.LinearRegression()
    model.fit(x_data, y_data)
    #print(model)
    #print(y_data)

    joblib.dump(model,output)
    # 测试
    # x_test = [[2, 0]]
    # predict = model.predict(x_test)
    # print("predict:", predict)
    #
    # ax = plt.figure().add_subplot(111, projection='3d')
    # ax.scatter(x_data[:, 0], x_data[:, 1], y_data, c='r', marker='o', s=100)  # 点为红色三角形
    # x0 = x_data[:, 0]
    # x1 = x_data[:, 1]
    # # 生成网格矩阵
    # print(model.coef_[0])
    # x0, x1 = np.meshgrid(x0, x1)
    # z = model.intercept_ + x0 * model.coef_[0] + x1 * model.coef_[1]
    # # # 画3D图
    # ax.plot_surface(x0, x1, z)
    # # # 设置坐标轴
    # ax.set_xlabel('Miles')
    # ax.set_ylabel('Num of Deliveries')
    # ax.set_zlabel('Time')
    # # 显示图像
    # plt.show()