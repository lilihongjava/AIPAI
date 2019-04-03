import numpy as np

#特殊处理
def sql_conver(input,output):
    data = np.genfromtxt(input)
    temp = data[:,2]
    for i in range(len(temp)):
        if temp[i] > 100 :
            temp[i] = 1
        else:
            temp[i] = 0
    #print(data)
    np.savetxt(output, data,fmt="%s")