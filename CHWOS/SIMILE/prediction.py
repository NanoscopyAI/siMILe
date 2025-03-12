from collections import defaultdict

import numpy as np  # type: ignore

from CHWOS.utils.log import get_logger

logger = get_logger(__name__)


def get_prediction_dict(bags, print_str=None):
    prediction_dict = defaultdict(list)
    inst = np.array([inst for bag in bags for inst in bag])

    for i in range(inst.shape[0]):
        prediction_dict[tuple(inst[i, :])].append(0)

    if print_str:
        logger.debug(f"{print_str} instance size: {inst.shape}")

    return prediction_dict
