import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, classification_report


def prediction(feature_column,input,input1,output):

    model = joblib.load(input1)
    # 测试
    rawdata = np.genfromtxt(input)
    #一维要处理
    if rawdata.ndim == 1:
        data = rawdata[np.newaxis,:]
    df = pd.DataFrame(rawdata)
    #print(df)
    data = np.array(df.iloc[:, feature_column])
    #print(data)
    predict = model.predict(data)
    print("predict:", predict)
    out = np.column_stack((rawdata, predict[:,np.newaxis]))
    np.savetxt(output, out,fmt="%s")
