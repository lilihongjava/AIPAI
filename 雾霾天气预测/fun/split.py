import numpy as np
import pandas as pd

#按比例拆分
def split(split_ratio,input,output,output1,random_seed_num=None):
    data = np.genfromtxt(input)
    df = pd.DataFrame(data)
    df1 = pd.DataFrame(data)
    df = df.sample(frac=split_ratio, random_state=random_seed_num)

    np.savetxt(output, df.values,fmt="%s")
    df1 = df1.sample(frac=1-split_ratio, random_state=random_seed_num)
    np.savetxt(output1, df1.values)