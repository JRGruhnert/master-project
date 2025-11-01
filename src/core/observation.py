from functools import cached_property

from tensordict import TensorDict
import torch


class BaseObservation(TensorDict):
    pass
