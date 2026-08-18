import jax
import optax
from jax import numpy as jp
from .custom_types import CustomTrainState, Transition

# This creates actor and critic train states containing their network parameters and optimizer states
def create_train_states(
    actor_network,
    critic_network, 
    input_dims, 
    n_actions, 
    critic_lr, 
    actor_lr, 
    key
):
    actor_key, critic_key = jax.random.split(key, 2)

    actor_dummy_obs     = jp.zeros((1, input_dims))
    critic_dummy_obs    = jp.zeros((1, input_dims))
    critic_dummy_action = jp.zeros((1, n_actions))
    actor_variables     = actor_network.init(actor_key, actor_dummy_obs)
    critic_variables    = critic_network.init(critic_key, critic_dummy_obs, critic_dummy_action)

    actor_train_state = CustomTrainState.create(
        apply_fn        = actor_network.apply,
        params          = actor_variables,
        target_params   = actor_variables,
        tx              = optax.adam(actor_lr),
    )

    critic_train_state = CustomTrainState.create(
        apply_fn        = critic_network.apply,
        params          = critic_variables,
        target_params   = critic_variables,
        tx              = optax.adam(critic_lr),
    )

    return actor_train_state, critic_train_state

def choose_actions(
    actor_train_state   : CustomTrainState,
    observations        : jp.ndarray,
    max_action          : float,
    min_action          : float,
    key                 : jax.Array,
    warmup              : bool = False
):
    actor_actions = actor_train_state.apply_fn(actor_train_state.params, observations)
    action_shape = actor_actions.shape
    key1, key2 = jax.random.split(key)

    # Warmup choose actions logic, pure noise
    def _warmup(_):
        actions = jax.random.uniform(key1, action_shape, minval=min_action, maxval=max_action)
        return actions

    # Normal choose actions logic, network with noise
    def _normal(_):
        noise               = jax.random.normal(key2, action_shape)
        actions_with_noise  = actor_actions + 0.1 * noise
        return jp.clip(actions_with_noise, min_action, max_action)

    actions = jax.lax.cond(warmup, _warmup, _normal, operand=None)
    return actions

# Choose deterministic actions
def choose_actions_deterministic(
    actor_train_state   : CustomTrainState,
    observations        : jp.ndarray,
    min_action          : float,
    max_action          : float,
):
    actor_actions = actor_train_state.apply_fn(actor_train_state.params, observations)
    return jp.clip(actor_actions, min_action, max_action)

def learn(
    actor_train_state   : CustomTrainState,
    critic_train_state  : CustomTrainState,
    discount            : float,
    data                : Transition,
):
    # Grab the dones and truncations from the Transition batch
    dones           = data.done
    truncations     = data.truncation
    truncation_mask = 1 - truncations
    done_mask       = 1 - dones

    # Double check brax wrapper environment logic but: 
    #   done = 1, truncation = 0 : true terminal
    #   done = 1, truncation = 1 : step limit reached

    # Calculate the target Q values
    next_actions = actor_train_state.apply_fn(actor_train_state.target_params, data.next_observation)                   # [B, A] 
    next_q_value = critic_train_state.apply_fn(critic_train_state.target_params, data.next_observation, next_actions)   # [B, 1]
    next_q_value = jp.squeeze(next_q_value, -1)                                                                         # [B]
    target_q     = jax.lax.stop_gradient(data.reward + discount * done_mask * next_q_value)                             # [B]

    # Calculate critic loss
    def critic_loss_fn(critic_params):
        # Calculate the predicted Q values
        pred_q = critic_train_state.apply_fn(critic_params, data.observation, data.action)                              # [B, 1]
        pred_q = jp.squeeze(pred_q,-1)                                                                                  # [B]
        error  = target_q - pred_q
        loss   = 0.5 * jp.mean(error * error * truncation_mask)
        return loss

    # Optimise the critic
    critic_loss, new_critic_grads = jax.value_and_grad(critic_loss_fn)(critic_train_state.params)
    critic_train_state = critic_train_state.apply_gradients(grads=new_critic_grads)

    # Calculate actor loss
    def actor_loss_fn(actor_params):
        # Actor loss is negative mean from the online critic
        predicted_actions = actor_train_state.apply_fn(actor_params, data.observation)                                  # [B, A]
        Q    = critic_train_state.apply_fn(critic_train_state.params, data.observation, predicted_actions)
        Q    = jp.squeeze(Q,-1)
        loss = -jp.mean(Q)
        return loss

    # Optimise the actor
    actor_loss, new_actor_grads = jax.value_and_grad(actor_loss_fn)(actor_train_state.params)
    actor_train_state = actor_train_state.apply_gradients(grads=new_actor_grads)

    return actor_train_state, critic_train_state

# Update frozen weights for target networks
def update_target_params(actor_state, critic_state, tau):
    new_actor_target_params = jax.tree_util.tree_map(
        lambda x, y: x * (1 - tau) + y * tau,
        actor_state.target_params,
        actor_state.params,
    )

    new_critic_target_params = jax.tree_util.tree_map(
        lambda x, y: x * (1 - tau) + y * tau,
        critic_state.target_params,
        critic_state.params,
    )
    actor_state = actor_state.replace(target_params=new_actor_target_params)
    critic_state = critic_state.replace(target_params=new_critic_target_params)
    return actor_state, critic_state
# def update_target_params(actor_state, critic_state, tau):
#     actor_state  = actor_state.replace(target_params=incremental_update(actor_state.params, actor_state.target_params, tau))
#     critic_state = critic_state.replace(target_params=incremental_update(critic_state.params, critic_state.target_params, tau))
#     return actor_state, critic_state
