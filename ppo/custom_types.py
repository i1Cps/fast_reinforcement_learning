from flax.training.train_state import TrainState
from flax import struct
from jax import numpy as jp
from typing import Any, Dict

@struct.dataclass
class Transition:
    observation     : jp.ndarray
    action          : jp.ndarray
    reward          : jp.ndarray
    done            : jp.ndarray
    truncation      : jp.ndarray
    next_observation: jp.ndarray
    raw_actions     : jp.ndarray
    log_probs       : jp.ndarray

@struct.dataclass
class RunningMeanStd:
    mean  : jp.ndarray
    var   : jp.ndarray
    count : jp.ndarray

@struct.dataclass
class CustomTrainState(TrainState):
    apply_fn: Dict[str, Any] = struct.field(pytree_node=False)
    normaliser_params: RunningMeanStd
