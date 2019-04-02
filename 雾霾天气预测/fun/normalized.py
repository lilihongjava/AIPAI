import numpy as np
import pandas as pd

def normalized(field,input, output):
    #np.set_printoptions(suppress=True)
    data = np.genfromtxt(input)
    df = pd.DataFrame(data)
    # print(df)
    for i in field:
        data[:, i] = Normalization(data[:,i])
    np.savetxt(output, data ,fmt="%s")

def Normalization(x):
    return [(float(i)-np.min(x))/float(max(x)-min(x)) for i in x]

