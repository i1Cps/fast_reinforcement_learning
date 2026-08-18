import jax
from jax import numpy as jp
from .custom_types import CustomTrainState, Transition
from .policy import choose_actions, choose_actions_deterministic
from functools import partial
from gymnax import EnvState
import gymnasium as gym
import mediapy as media

def env_step(
    train_state : CustomTrainState,
    env_states  : EnvState,
    epsilon     : float,
    key         : jax.Array,
    get_obs,
    step_env,
):

    action_key, step_key = jax.random.split(key, 2)

    observation = get_obs(env_states, None, None)

    actions = choose_actions(
        train_state   = train_state,
        observations  = observation,
        epsilon       = epsilon,
        key           = action_key,
    )

    next_observation, next_env_states, reward, done, _ = step_env(step_key, env_states, actions)

    return next_env_states, Transition(
        observation      = observation,
        action           = actions,
        reward           = reward,
        done             = done,
        next_observation = next_observation,
    )

# Evaluate the policy and return the average reward across environments
@partial(jax.jit, static_argnames=("reset","step_env", "get_obs","num_envs", "max_steps"))
def evaluate_policy(
    train_state : CustomTrainState,
    key         : jax.Array,
    reset,
    step_env,
    get_obs,
    num_envs    : int  = 128,
    max_steps   : int  = 500
):

    env_key, step_key = jax.random.split(key)

    # Reset all parallel envs
    env_keys   = jax.random.split(env_key, num_envs)
    obs, env_states = reset(env_keys, None)
    finished   = jp.zeros(num_envs, dtype=jp.bool)

    def step_fn(carry, _):
        env_states, finished = carry

        actions = choose_actions_deterministic(
            train_state  = train_state,
            observations = get_obs(env_states, None, None)
        )

        next_observation, next_env_states, reward, done, _ = step_env(step_key, env_states, actions)

        all_rewards = jp.where(finished, 0.0, reward)
        finished    = jp.logical_or(done, finished)

        return (next_env_states, finished), all_rewards

    (_, _), rewards = jax.lax.scan(
        f      = step_fn,
        init   = (env_states, finished),
        xs     = None,
        length = max_steps,
    )

    return jp.mean(jp.sum(rewards, axis=0))



def visualise_policy(
    train_state,
    env_name: str,
    rollout_length: int = 500,
    seed: int = 0,
):
    env = gym.make(env_name, render_mode="rgb_array")

    obs, _ = env.reset(seed=seed)

    frames = [env.render()]

    for _ in range(rollout_length):
        # Convert Gymnasium obs -> JAX batched obs for your policy
        batched_obs = jp.expand_dims(jp.asarray(obs), 0)

        # Deterministic policy action
        action = int(jax.device_get(choose_actions_deterministic(train_state, batched_obs)[0]))

        # Step Gymnasium env
        obs, reward, terminated, truncated, _ = env.step(action)

        # Capture frame
        frames.append(env.render())

        if terminated or truncated:
            break

    env.close()

    media.write_video("dqn/visual.mp4", frames, fps=30)
    print("Saved rollout to visual.mp4")
