import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib

def prediction(input,model,feature_column,output):

    model = joblib.load(model)
    # 测试
    data = np.genfromtxt(input)
    #一维要处理
    if data.ndim == 1:
        data = data[np.newaxis,:]
    df = pd.DataFrame(data)
    #print(df)
    data = np.array(df.iloc[:, feature_column])
    #print(data)
    x_test = [[3, 0]]
    predict = model.predict(data)
    print("predict:", predict)
    np.savetxt(output, predict)
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