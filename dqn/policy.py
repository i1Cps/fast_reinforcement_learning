import jax
import optax
from jax import numpy as jp
from .custom_types import Transition
from typing import Tuple

from .custom_types import CustomTrainState
from .network import QNetwork

# This creates a train state containing network parameters, optimizer state, and the network apply function
def create_train_state(
    network       : QNetwork,
    input_dims    : Tuple,
    learning_rate : float,
    key           : jax.Array
):

    dummy_obs  = jp.zeros((1, input_dims))
    variables  = network.init(key, dummy_obs)

    train_state = CustomTrainState.create(
        apply_fn      = network.apply,
        params        = variables,
        target_params = variables,
        tx            = optax.adam(learning_rate)
    )
    return train_state


def choose_actions(
    train_state   : CustomTrainState,
    observations  : jp.ndarray,
    epsilon       : float,
    key           : jax.Array,
):
    action_values = train_state.apply_fn(train_state.params, observations)
    num_actions   = action_values.shape[1]
    num_envs      = action_values.shape[0]
    exploration_key, action_key = jax.random.split(key)

    # Epsilon-greedy action selection
    explore_mask  = jax.random.uniform(exploration_key, (num_envs,)) < epsilon

    # Exploit: select the action with the highest Q-value
    greedy_actions = jp.argmax(action_values,axis=1)

    # Explore: select a random action
    random_actions = jax.random.randint(action_key, (num_envs,), 0, num_actions)

    actions = jp.where(explore_mask, random_actions, greedy_actions)
    return actions


def choose_actions_deterministic(
    train_state  : CustomTrainState,
    observations : jp.ndarray,
):
    action_values = train_state.apply_fn(train_state.params,observations)
    return jp.argmax(action_values, axis=1)


def learn(
    train_state : CustomTrainState,
    discount    : float,
    data        : Transition
):

    # Grab the dones from the Transition batch (gymnax only exposes a combined terminal)
    dones           = data.done
    done_mask       = 1 - dones

    # Calculate the target Q values
    next_q_values = train_state.apply_fn(train_state.target_params, data.next_observation)  # [B, A]
    max_next_q    = jp.max(next_q_values, axis=1)                                           # [B]
    target_q      = jax.lax.stop_gradient(data.reward + discount * max_next_q * done_mask)  # [B]

    def loss_fn(params):

        # Calculate the predicted Q values
        action_values = train_state.apply_fn(params, data.observation)                      # [B, A]
        chosen_actions = data.action[:, None]                                              # [B,1]
        chosen_actions_int = jp.asarray(chosen_actions, dtype=jp.int32)

        # Select the action value from the action in the buffer
        pred_q = jp.take_along_axis(action_values, chosen_actions_int, axis=1).squeeze(-1) # [B]
        error  = target_q - pred_q
        loss   = jp.mean(error * error)
        return loss

    # Optimise the Q network
    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grads)
    return new_train_state

def update_target_params(train_state: CustomTrainState) -> CustomTrainState:
    return train_state.replace(target_params=train_state.params)

def epsilon_decay(epsilon:float, decay_rate=0.9, epsilon_min= 0.01)->jp.ndarray:
    return jp.maximum(epsilon_min, epsilon * decay_rate)
