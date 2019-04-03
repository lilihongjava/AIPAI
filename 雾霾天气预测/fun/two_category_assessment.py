import numpy as np

def two_category_assessment(original_label_column,input,output,output1,output2):
    data = np.genfromtxt(input)
    total_samples = len(data)
    print("===========")
    y = data[:,-1]
    #正样本
    positive_samples = np.sum(y==1)
    negative_Samples = np.sum(y == 0)
    print(positive_samples)
    print(negative_Samples)

    np.savetxt(output,data)

