import jax
from jax import numpy as jp
from brax import envs
from .custom_types import CustomTrainState, Transition
from .policy import choose_actions, choose_actions_deterministic
from functools import partial
import mediapy as media

def env_step(
    actor_state : CustomTrainState,
    env_states  : envs.State,
    warmup      : bool,
    key         : jax.Array,
    env         : envs.Env
):

    actions = choose_actions(
        actor_train_state  = actor_state,
        observations       = env_states.obs,
        min_action         = -1,
        max_action         = 1,
        warmup             = warmup,
        key                = key,
    )

    next_env_states = env.step(env_states, actions)

    return next_env_states, Transition(
        observation      = env_states.obs,
        action           = actions,
        reward           = next_env_states.reward,
        done             = next_env_states.done,
        truncation       = next_env_states.info["truncation"],
        next_observation = next_env_states.obs,
    )

# Evaluate the policy and return the average reward across environments
@partial(jax.jit, static_argnames=("env", "num_envs", "max_steps"))
def evaluate_policy(
    actor_state : CustomTrainState,
    env         : envs.Env,
    key         : jax.Array,
    num_envs    : int  = 128,
    max_steps   : int  = 500
):

    env_keys   = jax.random.split(key, num_envs)
    env_states = env.reset(env_keys)
    finished   = jp.zeros(num_envs, dtype=jp.bool)

    def step_fn(carry, _):
        env_states, finished = carry

        actions = choose_actions_deterministic(
            actor_train_state = actor_state,
            observations      = env_states.obs,
            max_action        = 1,
            min_action        = -1,
        )

        next_env_states = env.step(env_states, actions)

        all_rewards = jp.where(finished, 0.0, next_env_states.reward)
        finished    = jp.logical_or(next_env_states.done, finished)

        return (next_env_states, finished), all_rewards

    (_, _), rewards = jax.lax.scan(
        f      = step_fn,
        init   = (env_states, finished),
        xs     = None,
        length = max_steps,
    )

    return jp.mean(jp.sum(rewards, axis=0))


def visualise_policy(
    actor_state : CustomTrainState, 
    env         : envs.Env, 
    key         : jax.Array
):

    rollout_length = 1000

    def do_rollout(state):
        def step_fn(state, _):
            action = choose_actions_deterministic(
                actor_train_state = actor_state,
                observations      = state.obs,
                max_action        = 1.0,
                min_action        = -1.0
            )
            next_state = env.step(state, action)
            return next_state, next_state

        _, traj = jax.lax.scan(
            f      = step_fn,
            init   = state,
            length = rollout_length,
            xs     = None
        )
        return traj

    reset_state = jax.jit(env.reset)(key)
    rollout     = jax.jit(do_rollout)(reset_state)

    # Extract the PyTree to a list
    rollout_list = [jax.tree.map(lambda x, j=j: x[j], rollout) for j in range(rollout_length)]

    # Render and save the rollout.
    render_every = 2
    fps          = 1.0 / env.dt / render_every
    print(f"FPS for rendering: {fps}")

    final_rollout = rollout_list[::render_every]
    pipeline_rollout = [state.pipeline_state for state in final_rollout]

    # Render the frames from each collected state in the rollout
    frames       = env.render(
        pipeline_rollout,
        width        = 640,
        height       = 480,
        camera       = "track",
    )
    media.write_video("ddpg/visual.mp4", frames, fps=fps)
    print("Saved rollout to visual.mp4")
