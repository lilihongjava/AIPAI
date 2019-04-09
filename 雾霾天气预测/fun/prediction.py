import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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
    print("predict:", predict)
    print("true_labels:", rawdata[:, 2])
    scores = model.predict_proba(data)[::, 1]

    print("分数:", scores)
    #print("auc1:", roc_auc_score(predict, model.decision_function(data)))
    fpr, tpr, thresholds = roc_curve(rawdata[:,2],  scores, pos_label=1)
    print("fpr!:", fpr)
    print("tpr!:", tpr)
    print("thresholds!:",thresholds)
    roc_auc = auc(fpr, tpr)
    print("roc_auc:",roc_auc)
    ## 绘制roc曲线图
    fpr, tpr, thresholds = roc_curve(rawdata[:, 2], scores, pos_label=1)
    print("fpr!:", fpr)
    print("tpr!:", tpr)
    print("thresholds!:", thresholds)
    roc_auc = auc(fpr, tpr)
    print("roc_auc:", roc_auc)
    ## 绘制roc曲线图
    poly = np.polyfit(fpr, tpr, 2)
    d = np.polyval(poly, fpr)
    plt.plot(fpr, tpr)
    #plt.plot(fpr, d,"r")
    plt.savefig('./test2.jpg')
    plt.show()

    out = np.column_stack((rawdata, predict[:,np.newaxis]))
    np.savetxt(output, out,fmt="%s")
