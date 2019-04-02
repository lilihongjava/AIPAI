import numpy as np

def logistic_regression(input,output):
    # 载入数据
    data = np.genfromtxt(input, delimiter=",")
    #print(data)
    np.savetxt(output,data)






