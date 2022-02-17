import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def summerize_history(history_dict):
    mat = []
    for key in history_dict.keys():
        if mat == []:
            mat = np.array(history_dict[key]).reshape(1,-1)
        else:
            mat = np.concatenate([mat, np.array(history_dict[key]).reshape(1,-1)], axis=0)
    return mat


# name = 'test_history_dict'
name = 'control_batch64_fc128-128'

res_dict = np.load(name+'.npy', allow_pickle=True).item()
reg = summerize_history(res_dict)            
plt.plot(np.mean(reg,axis = 0))
plt.savefig('test.png')

