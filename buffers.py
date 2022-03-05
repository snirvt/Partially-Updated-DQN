import collections

def get_exp_buffer():
    Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
    return Experience

def get_priorotized_exp_buffer():
    Priorotized_Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state', 'td_error'])
    return Priorotized_Experience