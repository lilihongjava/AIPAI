# encoding: utf-8
"""
@author: lee
@desc:
"""
import numpy as np
import pandas as pd
from sklearn.externals import joblib

def prediction(feature_column,input,input1,output):

    model = joblib.load(input1)
    # 测试
    rawdata = np.genfromtxt(input)
    #一维要处理
    if rawdata.ndim == 1:
        data = rawdata[np.newaxis,:]
    df = pd.DataFrame(rawdata)
    data = np.array(df.iloc[:, feature_column])
    predict = model.predict(data)
    #print("predict:", predict)
    #print("true_labels:", rawdata[:, 2])
    scores = model.predict_proba(data)[::, 1]

    out = np.column_stack((rawdata, predict[:,np.newaxis]))
    out = np.column_stack((out, scores[:, np.newaxis]))
    np.savetxt(output, out,fmt="%s")
