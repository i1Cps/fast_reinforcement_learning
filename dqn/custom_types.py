from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict
from flax import struct
from jax import numpy as jp

@struct.dataclass
class CustomTrainState(TrainState):
    target_params: FrozenDict

@struct.dataclass
class JaxReplayBuffer:
    data    : jp.ndarray
    ptr     : jp.ndarray
    capacity: jp.ndarray

@struct.dataclass
class Transition:
    observation     : jp.ndarray
    action          : jp.ndarray
    reward          : jp.ndarray
    done            : jp.ndarray
    next_observation: jp.ndarray
