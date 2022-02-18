import numpy as np
import torch


def init_hook(grad, p, minimum):
    group_size = grad.shape[0]*grad.shape[1]
    sub_group_size = max(minimum, int(group_size * p))
    return group_size, sub_group_size

def masking_grad(grad, idx):
    gradient_mask = torch.zeros_like(grad)
    row, col = np.unravel_index(idx, grad.shape)
    gradient_mask[row,col] = 1
    grad.requires_grad = False # allow in-place operation of leaf variable
    grad.mul_(gradient_mask)
    grad.requires_grad = True
    return grad


def rand_hook(grad, p=0.1, minimum = 1):
    group_size, sub_group_size = init_hook(grad, p, minimum) 
    idx = np.random.choice(group_size, size=sub_group_size, replace=0)
    return masking_grad(grad, idx)

def weighted_rand_hook(grad, p=0.1, minimum = 1):
    group_size, sub_group_size = init_hook(grad, p, minimum) 
    grad_abs = np.abs(grad.ravel().detach().numpy())
    weights = grad_abs/grad_abs.sum()
    idx = np.random.choice(group_size, size=sub_group_size, replace=0, p=weights)
    return masking_grad(grad, idx)

def greedy_hook(grad, p=0.1, minimum = 1):
    group_size, sub_group_size = init_hook(grad, p, minimum) 
    grad_abs = np.abs(grad.ravel().detach().numpy())
    idx = np.argsort(grad_abs)[-sub_group_size:] # greedy
    return masking_grad(grad, idx)

def epsilon_greedy_hook(grad, p=0.1, epsilon = 0.5, minimum = 1):
    group_size, sub_group_size = init_hook(grad, p, minimum) 
    grad_abs = np.abs(grad.ravel().detach().numpy())
    idx_greedy = np.argsort(grad_abs)[-int(np.ceil(sub_group_size*epsilon)):] # greedy
    available_values = set(range(group_size)) - set(idx_greedy)
    idx_rand = np.random.choice(list(available_values),                         # rand
                                size=int(np.ceil(sub_group_size*(1-epsilon))),
                                 replace=0)
    idx = np.append(idx_greedy, idx_rand)
    return masking_grad(grad, idx)
