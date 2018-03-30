import contextlib
from enum import Enum
import json
import logging
import os
import random
import string

from gym import spaces
import numpy as np
import ruamel.yaml as yaml

from .envs import utility


def random_string(num_chars):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(num_chars))


def save_config(config, logdir=None):
    """Save a new configuration by name.

    If a logging directory is specified, is will be created and the configuration
    will be stored there. Otherwise, a log message will be printed.

    Args:
      config: Configuration object.
      logdir: Location for writing summaries and checkpoints if specified.

    Returns:
      Configuration object.
    """
    if logdir:
        with config.unlocked:
            config.logdir = logdir
        message = 'Start a new run and write summaries and checkpoints to {}.'
        logging.info(message.format(config.logdir))
        os.makedirs(config.logdir)
        config_path = os.path.join(config.logdir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        message = (
            'Start a new run without storing summaries and checkpoints since no '
            'logging directory was specified.')
        logging.info(message)
    return config


def load_config(logdir):
    """Load a configuration from the log directory.

    Args:
      logdir: The logging directory containing the configuration file.

    Raises:
      IOError: The logging directory does not contain a configuration file.

    Returns:
      Configuration object.
    """
    config_path = logdir and os.path.join(logdir, 'config.yaml')
    if not config_path or not os.path.exists(config_path):
        message = (
            'Cannot resume an existing run since the logging directory does not '
            'contain a configuration file.')
        raise IOError(message)
    with open(config_path, 'r') as file_:
        config = yaml.load(file_)
    message = 'Resume run and write summaries and checkpoints to {}.'
    logging.info(message.format(config.logdir))
    return config


class AttrDict(dict):
    """Wrap a dictionary to access keys as attributes."""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        super(AttrDict, self).__setattr__('_mutable', False)

    def __getattr__(self, key):
        # Do not provide None for unimplemented magic attributes.
        if key.startswith('__'):
            raise AttributeError
        return self.get(key, None)

    def __setattr__(self, key, value):
        if not self._mutable:
            message = "Cannot set attribute '{}'.".format(key)
            message += " Use 'with obj.unlocked:' scope to set attributes."
            raise RuntimeError(message)
        if key.startswith('__'):
            raise AttributeError("Cannot set magic attribute '{}'".format(key))
        self[key] = value

    @property
    @contextlib.contextmanager
    def unlocked(self):
        super(AttrDict, self).__setattr__('_mutable', True)
        yield
        super(AttrDict, self).__setattr__('_mutable', False)

    def copy(self):
        return type(self)(super(AttrDict, self).copy())


class PommermanJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, utility.Item):
            return obj.value
        elif isinstance(obj, utility.Action):
            return obj.value
        elif isinstance(obj, np.int64):
            return int(obj)
        elif hasattr(obj, 'to_json'):
            return obj.to_json()
        elif isinstance(obj, spaces.Discrete):
            return obj.n
        elif isinstance(obj, spaces.Tuple):
            return [space.n for space in obj.spaces]
        return json.JSONEncoder.default(self, obj)
