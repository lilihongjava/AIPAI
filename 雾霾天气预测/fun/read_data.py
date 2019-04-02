import numpy as np

def read_data(input,output):
    # 载入数据
    data = np.genfromtxt(input, delimiter=",")
    #print(data)
    np.savetxt(output,data)






