import flax.linen as nn 
from jax import numpy as jp
from typing import Callable, Sequence

# Create the network
class QNetwork(nn.Module):
    h_dims        : Sequence[int]
    num_actions   : int
    activation_fn : Callable = nn.relu

    @nn.compact
    def __call__(self, state: jp.ndarray) -> jp.ndarray:
        for h in self.h_dims:
            state = nn.Dense(h)(state)
            state = self.activation_fn(state)

        # Returns an estimate of the expected return for each possible action
        action_values = nn.Dense(self.num_actions)(state)
        return action_values
