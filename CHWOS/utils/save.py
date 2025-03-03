import os
import numpy as np # type: ignore
from CHWOS.utils.log import get_logger
from CHWOS.utils.misc import get_date_str, description_dict_to_str

logger = get_logger(__name__)

def get_savefile_name(save_name_dict, include_date=True):
    name = description_dict_to_str(save_name_dict)
    if include_date:
        date_str = get_date_str()
        name += date_str
    return name


def save_data_numpy(data, save_location, name, iteration=None):
    save_path = os.path.join(save_location, name + '.npy')
    logger.debug(f'Saving to: {save_path}')
    np.save(save_path, data)