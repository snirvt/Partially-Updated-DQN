import numpy as np
import torch

def rand_hook(grad, p=0.1, minimum = 1):
    gradient_mask = torch.zeros_like(grad)
    group_size = grad.shape[0]*grad.shape[1]
    sub_group_size = max(minimum, int(group_size * p)) 
    idx = np.random.choice(group_size, size=sub_group_size, replace=0)
    row, col = np.unravel_index(idx,grad.shape)
    gradient_mask[row,col] = 1
    grad.mul_(gradient_mask)
    return grad




