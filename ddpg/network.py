from typing import Callable, Sequence
from flax import linen as nn
from jax import numpy as jp

# This network corresponds to our value function, it estimates the expected return for a state action pair
class CriticNetwork(nn.Module):
    h_dims          : Sequence[int]
    activation_fn   : Callable = nn.relu

    @nn.compact 
    def __call__(self, state, actions):
        state_action = jp.concat([state, actions], axis=1)
        for h in self.h_dims:
            state_action = nn.Dense(h)(state_action)
            state_action = self.activation_fn(state_action)

        # Final output layer
        state_action_value = nn.Dense(1)(state_action)
        return state_action_value


# This network corresponds to our policy, it chooses actions given a state
class ActorNetwork(nn.Module):
    h_dims          : Sequence[int] 
    n_actions       : int
    activation_fn   : Callable = nn.relu

    @nn.compact
    def __call__(self, state):
        for h in self.h_dims:
            state = nn.Dense(h)(state)
            state = self.activation_fn(state)

        # Final output layer
        actions = nn.Dense(self.n_actions)(state)
        actions = nn.tanh(actions)
        return actions


