import random
import numpy as np
import pandas as pd


def random_sampling(input=None,num_of_samples=None,sampling_ratio=None,replace=None,random_seed_num=None,output=None):
    data = np.genfromtxt(input)
    #print(data)
    df = pd.DataFrame(data)
    if num_of_samples is not None:
        df = df.sample(n=num_of_samples, replace = replace,random_state=random_seed_num)
    elif sampling_ratio is not None:
        df = df.sample(frac=sampling_ratio, replace = replace, random_state=random_seed_num)
    np.savetxt(output, df.values)
