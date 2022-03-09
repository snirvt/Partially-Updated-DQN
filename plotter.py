from cProfile import label
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


''' random hook '''
title = 'Equal Probabilities per Gradient'
# headers = [
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_rand_grad_1_075','1_075'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_rand_grad_1_05','1_05'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_rand_grad_1_025','1_025'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_rand_grad_075_075','075_075'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_rand_grad_075_05','075_05'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_rand_grad_075_025','075_025'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_rand_grad_05_075','05_075'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_rand_grad_05_05','05_05'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_rand_grad_05_025','05_025'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_rand_grad_025_075','025_075'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_rand_grad_025_05','025_05'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_rand_grad_025_025','025_025'),
# ]


'''2 layers''' 
title = '2 layers'
# headers = [('control_small_dqn_2frames_lr0.0001_sync1','sync_1'),
#            ('control_small_dqn_2frames_lr0.0001_sync500','sync_500'),
#            ('small_dqn_2frames_lr0.0001_2_layers_only_rand_hook_grad_1_09_1_09', 'rand_1_09_1_09'),
#            ('small_dqn_2frames_lr0.0001_2_layers_only_rand_hook_grad_1_08_1_08', 'rand_1_08_1_08'),
#            ('small_dqn_2frames_lr0.0001_2_layers_only_weighted_rand_hook_grad_1_09_1_09', 'weighted_1_09_1_09'),
#            ('small_dqn_2frames_lr0.0001_2_layers_only_weighted_rand_hook_grad_1_08_1_08', 'weighted_1_08_1_08'),
#            ('small_dqn_2frames_lr0.0001_2_layers_only_greedy_hook_grad_1_09_1_09', 'greedy_1_09_1_09'),
#            ('small_dqn_2frames_lr0.0001_2_layers_only_greedy_hook_grad_1_08_1_08', 'greedy_1_08_1_08'),
# ]



''' weighted hook'''
# title = 'Weighted Probabilities Based on Gradient\n Magnitude for Keeping Gradients'
# headers=[
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_weighted_rand_grad_1_075', '1_075'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_weighted_rand_grad_1_05', '1_05'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_weighted_rand_grad_1_025', '1_025'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_weighted_rand_grad_075_075', '075_075'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_weighted_rand_grad_075_05', '075_05'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_weighted_rand_grad_075_025', '075_025'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_weighted_rand_grad_05_075', '05_075'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_weighted_rand_grad_05_05', '05_05'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_weighted_rand_grad_05_025', '05_025'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_weighted_rand_grad_025_075', '025_075'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_weighted_rand_grad_025_05', '025_05'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_weighted_rand_grad_025_025', '025_025'),
# ]

''' greedy hook'''
# title = 'Choosing Gradients Greedily'
# headers = [('small_dqn_2frames_lr0.0001_first(real)_layer_only_greedy_hook_grad_1_075','1_075'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_greedy_hook_grad_1_05','1_05'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_greedy_hook_grad_1_025','1_025'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_greedy_hook_grad_075_075','075_075'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_greedy_hook_grad_075_05','075_05'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_greedy_hook_grad_075_025','075_025'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_greedy_hook_grad_05_075','05_075'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_greedy_hook_grad_05_05','05_05'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_greedy_hook_grad_05_025','05_025'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_greedy_hook_grad_025_075','025_075'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_greedy_hook_grad_025_05','025_05'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_greedy_hook_grad_025_025','025_025'),
# ]


''' epsilon greedy hook'''
# title = 'Choosing Gradients Epsilon Greedily'
# headers = [('small_dqn_2frames_lr0.0001_first(real)_layer_only_epsilon_greedy_hook_grad_1_075_05','1_075_05'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_epsilon_greedy_hook_grad_1_05_05','1_05_05'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_epsilon_greedy_hook_grad_1_025_05','1_025_05'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_epsilon_greedy_hook_grad_1_075_09','1_075_09'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_epsilon_greedy_hook_grad_1_05_09','1_05_09'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_epsilon_greedy_hook_grad_1_025_09','1_025_09'),
# ]


''' control '''
# title = 'Control'
# headers = [('control_small_dqn_2frames_lr0.0001_sync1000','sync_1000'),
#            ('control_small_dqn_2frames_lr0.0001_sync500','sync_500'),
#            ('control_small_dqn_2frames_lr0.0001_sync250','sync_250'),
#            ('control_small_dqn_2frames_lr0.0001_sync125', 'sync_125'),
#            ('control_small_dqn_2frames_lr0.0001_sync62', 'sync_62'),
#            ('control_small_dqn_2frames_lr0.0001_sync1', 'sync_1'),
#           ]


'''Last Comparison'''
# title = 'Last Comparison'
# headers = [('control_small_dqn_2frames_lr0.0001_sync1','sync_1'),
#            ('control_small_dqn_2frames_lr0.0001_sync500','sync_500'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_rand_grad_1_075','rand_1_075'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_rand_grad_05_075','rand_05_075'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_weighted_rand_grad_1_075','weighted_1_075'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_greedy_hook_grad_075_075','greedy_075_075'),
#            ('small_dqn_2frames_lr0.0001_first(real)_layer_only_greedy_hook_grad_05_075','greedy_05_075'),
# ]


'''Prioritized'''
title = 'Prioritized'
headers = [('control_small_dqn_2frames_lr0.0001_sync1_prioratized','prioritized_sync_1'),
           ('control_small_dqn_2frames_lr0.0001_sync1', 'sync_1'),
           ('control_small_dqn_2frames_lr0.0001_sync1000_prioratized','prioritized_sync_1000'),
           ('control_small_dqn_2frames_lr0.0001_sync1000','sync_1000'),
           #('small_dqn_2frames_lr0.0001_first_layer_only_rand_hook_grad_1_075_priorazited','prioritized_1_075'),
           #('small_dqn_2frames_lr0.0001_first_layer_only_rand_hook_grad_1_05_priorazited','prioritized_1_05'),
]

for head in headers:
    head_dict = np.load('results/'+head[0]+'.npy', allow_pickle=True).item()
    head_history = summerize_history(head_dict)
    plt.plot(np.mean(head_history, axis = 0), label = head[1])
plt.xlabel('# Games')
plt.ylabel('Reward')
plt.title('Pong - {}'.format(title))
plt.legend()
plt.savefig('plot.png')




from matplotlib.pyplot import figure
figure(figsize=(8, 8), dpi=80)

data = []
labels = []
for head in headers:
    head_dict = np.load('results/'+head[0]+'.npy', allow_pickle=True).item()
    head_history = summerize_history(head_dict)
    data.append(np.mean(head_history[:,-50:], axis = 0))
    labels.append(head[1])
plt.boxplot(data)
plt.xticks(list(range(1,len(labels)+1)),labels, rotation=20, fontsize=12)
plt.yticks(fontsize=20)
plt.ylabel('Reward', fontsize=16)
plt.title('Pong - {}'.format(title), fontsize=20)
plt.savefig('boxplot.png')
plt.show()


