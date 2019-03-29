import numpy as np
import math

def missing_value_padding(input,filled_field,replace_with,output):
    #data = np.genfromtxt("job.csv", delimiter=",")
    data = np.genfromtxt(input)
    filled_field_data = filled_field.split(",")
    #print(data)
    #print(len(data))#几行
    #print(len(data[0]))#几列
    #把原值替换数值max min mean
    for i in range(len(filled_field_data)):
        #i 从0开始
        if(i+1>len(data[0])) :
            break
        temp = data[:, i]
        test = np.array([3, 5, 4, 7, np.nan])
        #print(np.nanmax(test))
        replace("2", test)
        for j in range(len(temp)):
             if math.isnan(temp[j]) or temp[j] == '':
                 #temp[j] = np.nanmax(temp)
                 temp[j] = replace(replace_with, temp)
                 #print(data)
    np.savetxt(output, data)
    return  data

def replace(var,temp):
    return {
            '1': lambda temp: np.nanmin(temp),
            '2': lambda temp: np.nanmax(temp),
            '3': lambda temp: np.nanmean(temp),
    }[var](temp)