import numpy as np # type: ignore
from numpy.random import SeedSequence # type: ignore
from datetime import date


def combine_inst_dics(original, new):
    for k in new.keys():
        original[k].extend(new[k])

def set_random_and_get_sequence(seed):
    np.random.seed(seed)
    ss = SeedSequence(seed)
    return ss

def description_dict_to_str(save_name_dict):
    save_name_str = ''
    for i, (k, v) in enumerate(save_name_dict.items()):
        if i != 0:
            save_name_str += '-'
        save_name_str += '{}_{}'.format(k, v)
    return save_name_str

def get_date_str():
    date_raw = date.today()
    date_str = '-DMY_{}_{}_{}'.format(date_raw.day, date_raw.month, date_raw.year)
    return date_str