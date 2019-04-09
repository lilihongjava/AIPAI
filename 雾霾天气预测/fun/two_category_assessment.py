import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

def two_category_assessment(original_label_column,input,output,output1,output2):
    data = np.genfromtxt(input)
    total_samples = len(data)
    y = data[:, original_label_column]
    pred = data[:,-2]
    #正样本
    positive_samples = np.sum(y==1)
    negative_Samples = np.sum(y == 0)
    print(positive_samples)
    print(negative_Samples)

    fpr, tpr, thresholds = roc_curve(y, pred)
    print("y:", y)
    print("pred:", pred)
    print("fpr:",fpr)
    print("tpr:", fpr)

    print("thresholds:",thresholds)
    #print("auc1:", roc_auc_score(data[:, -2]),y)
    print("auc:", auc(fpr, tpr))

    np.savetxt(output,data)

