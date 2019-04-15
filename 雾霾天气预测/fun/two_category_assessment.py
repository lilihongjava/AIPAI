# encoding: utf-8
"""
@author: lee
@desc:
"""
import json
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def two_category_assessment(original_label_column, input0, output0):
    """
    :param original_label_column:
    :param input0:
    :param output0:
    :return:
    """
    data = np.genfromtxt(input0)
    y_data = data[:, original_label_column]
    scores = data[:, -1]
    print("分数:", scores)
    fpr, tpr, thresholds = roc_curve(y_data, scores, pos_label=1, drop_intermediate=False)
    print("fpr!:", fpr)
    print("tpr!:", tpr)
    print("thresholds!:", thresholds)
    roc_auc = auc(fpr, tpr)
    print("roc_auc:", roc_auc)
    roc_list = []
    for i, element in enumerate(fpr):
        python2json = dict()
        python2json["x"] = element
        python2json["y"] = tpr[i]
        roc_list.append(python2json)
    json_str = json.dumps(roc_list)
    print(json_str)
    # 绘制roc曲线图
    poly = np.polyfit(fpr, tpr, 2)
    poly_val = np.polyval(poly, fpr)
    plt.plot(fpr, tpr)
    plt.plot(fpr, poly_val, "r")
    # plt.savefig('./test2.jpg')
    plt.show()

    # #正样本
    # positive_samples = np.sum(y==1)
    # # 负样本
    # negative_Samples = np.sum(y == 0)
    # print(positive_samples)
    # print(negative_Samples)
    np.savetxt(output0, data)
