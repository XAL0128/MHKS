import argparse
import os
import torch
import random
import numpy as np
import torch.nn as nn
import logging


def seed(s):
    if isinstance(s, int):
        if 0 <= s <= 9999:
            return s
        else:
            raise argparse.ArgumentTypeError(
                "Seed must be between 0 and 2**32 - 1. Received {0}".format(s)
            )
    elif s == "random":
        return random.randint(0, 9999)
    else:
        raise argparse.ArgumentTypeError(
            "Integer value is expected. Recieved {0}".format(s)
        )


def set_random_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dict2detach(x):
    output = {}
    for key, value in x.items():
        output[key] = value.detach()

    return output


def init_weights(layer):
    """
    Initialize weights.
    """
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        if layer.bias is not None: layer.bias.data.zero_()


def set_requires_grad(nets, requires_grad=False):
    """
    Set requires_grad=False for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class TriggerPeriod(object):
    """
    Trigger using period:
        For the example 'period=10, trigger=3', then 0,1,2 (valid), 3,4,5,6,7,8,9 (invalid).
        For the example 'period=10, trigger=-3', then 0,1,2,3,4,5,6 (invalid), 7,8,9 (valid).
    """

    def __init__(self, period, area):
        assert period > 0
        # Get lambda & init
        self._lmd_trigger = (lambda n: n < area) if area >= 0 else (lambda n: n >= period + area)
        # Configs
        self._period = period
        self._count = 0

    def check(self):
        # 1. Get return
        ret = self._lmd_trigger(self._count)
        # 2. Update counts
        self._count = (self._count + 1) % self._period
        # Return
        return ret


class NumberTracker:
    def __init__(self):
        self.data = {}

    def add_number(self, name, value):
        if name in self.data:
            self.data[name].append(value)
        else:
            self.data[name] = [value]

    def calculate_stats(self):
        stats = {}
        for name, values in self.data.items():
            total = sum(values)
            count = len(values)
            if count > 0:
                average = total / count
            else:
                average = 0
            stats[name] = {
                "total": total,
                "average": average
            }
        return stats


def mkdirs(*args):
    for path in args:
        if not os.path.exists(path):
            os.makedirs(path)


def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " % (key, src_dict[key])
    return dst_str


def set_logger(log_dir, log_name, verbose_level=1, show=True):
    # base logger
    log_file_name = '{}.log'.format(log_name)
    log_file_path = os.path.join(log_dir, log_file_name)
    logger = logging.getLogger('MHKS')
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    if show:
        # stream handler
        stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
        ch = logging.StreamHandler()
        ch.setLevel(stream_level[verbose_level])
        ch_formatter = logging.Formatter('%(name)s - %(message)s')
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

    return logger
