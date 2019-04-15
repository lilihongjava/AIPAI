# encoding: utf-8
"""
@author: lee
@desc:
"""
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import numpy as np
import pandas as pd


def random_forest(feature_column, label_column, input0, output0):
    """
    :param feature_column:
    :param label_column:
    :param input0:
    :param output0:
    :return:
    """
    data = np.genfromtxt(input0)
    data_frame = pd.DataFrame(data)
    x_data = np.array(data_frame.iloc[:, feature_column])
    y_data = data[:, label_column]
    # plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
    # plt.show()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5)
    dtree = tree.DecisionTreeClassifier()
    dtree.fit(x_train, y_train)
    joblib.dump(dtree, output0)
