import numpy as np
import torch

def init_hook(grad, p, minimum):
    group_size = torch.numel(grad)
    sub_group_size = max(minimum, int(group_size * p))
    return group_size, sub_group_size

def init_hook_non_zero(grad, p_non_zero, p_all, minimum):
    group_size = torch.numel(grad)
    non_zero_idx = torch.where(grad.ravel()!=0)[0].cpu()
    non_zero_size = non_zero_idx.numel()
    size = min(int(non_zero_size * p_non_zero), int(group_size * p_all))
    sub_group_size = max(minimum, size)
    return non_zero_size, sub_group_size, non_zero_idx

def masking_grad(grad, idx):
    gradient_mask = torch.zeros_like(grad)
    tuple_idx = np.unravel_index(idx, grad.shape)
    gradient_mask[tuple_idx] = 1
    grad.requires_grad = False # allow in-place operation of leaf variable
    grad = grad.mul_(gradient_mask)
    grad.requires_grad = True
    return grad


def rand_hook(grad, p_non_zero = 1/2, p_all=1/4, minimum = 1):
    group_size, sub_group_size, non_zero_idx = init_hook_non_zero(grad, p_non_zero, p_all, minimum)
    try:
        idx = np.random.choice(non_zero_idx.squeeze(), size=sub_group_size, replace=0)
    except:
        print('rand_hook exception')
        return grad
    return masking_grad(grad, idx)

def weighted_rand_hook(grad, p_non_zero = 1/2, p_all=1/4, minimum = 1):
    group_size, sub_group_size, non_zero_idx = init_hook_non_zero(grad, p_non_zero, p_all, minimum)
    grad_abs = np.abs(grad.ravel().detach().cpu().numpy())
    weights = grad_abs[non_zero_idx]/grad_abs[non_zero_idx].sum()
    try:
        idx = np.random.choice(non_zero_idx.squeeze(), size=sub_group_size, replace=0, p=weights.squeeze())
    except:
        print('weighted_rand_hook exception')
        return grad
    return masking_grad(grad, idx)

def greedy_hook(grad, p_non_zero = 1/2, p_all=1/4, minimum = 1):
    group_size, sub_group_size, non_zero_idx = init_hook_non_zero(grad, p_non_zero, p_all, minimum)
    grad_abs = np.abs(grad.ravel()[non_zero_idx].detach().cpu().numpy())
    try:
        idx = np.argsort(grad_abs)[-sub_group_size:] # greedy
    except:
        print('greedy_hook exception')
        return grad
    return masking_grad(grad, non_zero_idx[idx])

def epsilon_greedy_hook(grad, p_non_zero=0.1, p_all = 0.1, epsilon = 0.5, minimum = 1):
    group_size, sub_group_size, non_zero_idx = init_hook_non_zero(grad, p_non_zero, p_all, minimum)
    grad_abs = np.abs(grad.ravel()[non_zero_idx].detach().cpu().numpy())
    try:
        idx_greedy = np.argsort(grad_abs)[-int(np.ceil(sub_group_size*(1-epsilon))):] # greedy
        available_values = set(range(group_size)) - set(idx_greedy)
        idx_rand = np.random.choice(list(available_values),                         # rand
                                    size=int(np.ceil(sub_group_size*epsilon)), replace=0)
        idx = np.append(idx_greedy, idx_rand)
    except:
        print('epsilon_greedy_hook exception')
        return grad
    return masking_grad(grad, idx)
