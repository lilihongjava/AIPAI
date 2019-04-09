import numpy as np
import pandas as pd

#按比例拆分
def split(split_ratio,input,output,output1,random_seed_num=None):
    data = np.genfromtxt(input)
    df = pd.DataFrame(data)
    #print("data:" + data)
    df1 = df.sample(frac=split_ratio, random_state=random_seed_num)
    np.savetxt(output, df1.values,fmt="%s")
    #print(df1.shape)
    df = df.append(df1)
    #求差集
    np.savetxt(output1, df.drop_duplicates(keep=False),fmt="%s")