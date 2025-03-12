# logger.py
import logging
import os
import sys

from CHWOS.utils.misc import description_dict_to_str, get_date_str

BASE_LOG_PATH = None


def get_logger(filename):
    if not BASE_LOG_PATH:
        raise ValueError("BASE_LOG_PATH not set")

    logger = logging.getLogger(filename)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(BASE_LOG_PATH)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(name)s][%(levelname)s]: %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def set_logger_path_with_config(exp_config):
    global BASE_LOG_PATH

    save_name_dict = {
        "SIMILE": exp_config.DATA_NAME,
        "A": exp_config.A_TAG,
        "B": exp_config.B_TAG,
        "AE": exp_config.AE,
        "SC": exp_config.SYM_C,
        "sig": exp_config.sigma,
        "c": exp_config.C,
        "bs": exp_config.bagsize,
        "fld": exp_config.fold_split_idx,
        "NCV_fold": exp_config.NCV_fold,
        "minacc": exp_config.MIN_ACC,
    }
    if exp_config.AE:
        del save_name_dict["AE"]
    if exp_config.SYM_C:
        del save_name_dict["SC"]
    if exp_config.fold_split_idx == 0:
        del save_name_dict["fld"]
    if exp_config.NCV_fold == 0:
        del save_name_dict["NCV_fold"]

    save_string = get_logfile_name(save_name_dict, include_date=True)
    BASE_LOG_PATH = os.path.join(exp_config.LOG_LOCATION, save_string + ".log")


def get_logfile_name(description_dict, include_date=True):
    name = description_dict_to_str(description_dict)
    if include_date:
        date_str = get_date_str()
        name += date_str
    return name
