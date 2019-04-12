# encoding: utf-8
"""
@author: lee
@desc:
"""
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import json
def two_category_assessment(original_label_column,input,output,output1,output2):
    data = np.genfromtxt(input)
    y = data[:, original_label_column]
    scores = data[:,-1]
    print("分数:", scores)
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1,drop_intermediate=False)
    print("fpr!:", fpr)
    print("tpr!:", tpr)
    print("thresholds!:", thresholds)
    roc_auc = auc(fpr, tpr)
    print("roc_auc:", roc_auc)
    list = []
    for i in range(len(fpr)):
        python2json = {}
        python2json["x"] = fpr[i]
        python2json["y"] = tpr[i]
        list.append(python2json)
    jsonStr = json.dumps(list)
    print(jsonStr)
    # 绘制roc曲线图
    # poly = np.polyfit(fpr, tpr, 2)
    # d = np.polyval(poly, fpr)
    # plt.plot(fpr, tpr)
    # plt.plot(fpr, d,"r")
    # plt.savefig('./test2.jpg')
    # plt.show()

    # #正样本
    # positive_samples = np.sum(y==1)
    # # 负样本
    # negative_Samples = np.sum(y == 0)
    # print(positive_samples)
    # print(negative_Samples)

    np.savetxt(output,data)

