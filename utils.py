import math
import yaml
import logging
import os
import inspect
import torch
import pickle


def save(lst, path):
    with open(path, 'wb') as fp:
        pickle.dump(lst, fp)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def config_log(log_dir, fname):
    logger = logging.getLogger('SAT')
    logger_formatter = logging.Formatter('[%(asctime)s]%(filename)s[line:%(lineno)d] [%(levelname)s]: %(message)s')

    logger_ch = logging.StreamHandler()
    logger_ch.setFormatter(logger_formatter)
    logger.addHandler(logger_ch)

    logger_fh = logging.FileHandler(os.path.join(log_dir, '{}.log'.format(fname)))
    logger_fh.setFormatter(logger_formatter)
    logger.addHandler(logger_fh)
    logger.setLevel(logging.INFO)
    return logger








