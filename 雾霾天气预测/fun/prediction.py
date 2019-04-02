import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib

def prediction(feature_column,input,input1,output):

    model = joblib.load(input1)
    # 测试
    data = np.genfromtxt(input)
    #一维要处理
    if data.ndim == 1:
        data = data[np.newaxis,:]
    df = pd.DataFrame(data)
    #print(df)
    data = np.array(df.iloc[:, feature_column])
    #print(data)
    predict = model.predict(data)
    print("predict:", predict)
    np.savetxt(output, predict)
