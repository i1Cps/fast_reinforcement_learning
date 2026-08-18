import jax
from jax import numpy as jp
from brax import envs
from .custom_types import CustomTrainState, Transition
from .policy import choose_actions
from functools import partial
import mediapy as media

# Step the parallel environments one time
def parallel_env_step(
    train_state: CustomTrainState,
    env_states  : envs.State,
    key         : jax.Array,
    env         : envs.Env
):

    next_key, current_key = jax.random.split(key, 2)

    actions, extra_info = choose_actions(
        observations = env_states.obs,
        train_state  = train_state,
        key          = current_key,
    )

    next_env_states = env.step(env_states, actions)

    return (
        Transition(
            observation      = env_states.obs,
            action           = actions,
            reward           = next_env_states.reward,
            done             = next_env_states.done,
            truncation       = next_env_states.info["truncation"],
            next_observation = next_env_states.obs,
            raw_actions      = extra_info["raw_actions"],
            log_probs        = extra_info["log_probs"]
        ),
        next_env_states,
        next_key
    )

# Returns a batch of trajectories of shape ~ (unroll_length, num_envs, data)
def unroll(train_state, env_states, key, env, unroll_length):
    # Scan function that follows haskell like signature
    def step_func(carry, _):
        env_states, key = carry
        transition, next_env_states, next_key = parallel_env_step(train_state, env_states, key, env)
        return (next_env_states, next_key), transition

    (next_states, next_key), generated_transitions = jax.lax.scan(
        f      = step_func,
        xs     = None,
        init   = (env_states, key),
        length = unroll_length
    )
    return generated_transitions, next_states, next_key 

# This function evaluates a policy and returns the average rewards across multiple environments
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

        actions, _ = choose_actions(
            train_state  = actor_state,
            observations = env_states.obs,
            key          = key,
            deterministic = True,
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
            action, _ = choose_actions(
                train_state  = actor_state,
                observations = state.obs,
                key          = key,
                deterministic = True,
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
    media.write_video("ppo/visual.mp4", frames, fps=fps)
    print("Saved rollout to ppo/visual.mp4")
