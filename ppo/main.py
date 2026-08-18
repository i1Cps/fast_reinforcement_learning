import argparse
import functools
import os
import time

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

import jax
from jax import numpy as jp
from brax import envs
from flax import linen

from . import normalise
from .actor import evaluate_policy, unroll, visualise_policy
from .custom_types import CustomTrainState
from .logger import log_eval
from .loss_function import ppo_loss_fn
from .network import ActorNetwork, CriticNetwork
from .policy import create_train_state, learn


def main(rng: int):
    # NOTE: PPO hyperparameters
    training_steps          = 50_000_000
    num_envs                = 4096
    unroll_length           = 10
    batch_size              = 512 # NOTE: interpret this as number of trajectories in each batch
    num_minibatches         = 32
    updates_per_batch       = 8
    learning_rate           = 0.0003

    discount                = 0.990
    entropy_cost            = 0.005
    gae_lambda              = 0.95
    clipping_epsilon        = 0.2
    vf_coefficient          = 0.5
    num_evals               = 10
    env_name                = 'humanoid'

    steps_per_eval = training_steps // num_evals
    steps_per_train_step = batch_size * num_minibatches * unroll_length
    steps_per_epoch = steps_per_eval // steps_per_train_step



    print(f"Welcome to jax rl, the functional way! \n\n \
    We're about to train our agent on the environment {env_name} using PPO \n")

    print(
        f"""Training Params: \n
        Training steps                        : {training_steps}
        Number of parallel environments       : {num_envs}
        Number of steps per unroll            : {unroll_length}
        Number of trajectories in each batch  : {batch_size}
        Number of mini-batches                : {num_minibatches}
        Number of updates per batch           : {updates_per_batch}
        Number of steps per epoch             : {steps_per_epoch}
        """
    )

    # Measure compilation time
    jit_start_time = time.time()

    # Prepare the parallel environments
    env = envs.get_environment(
        env_name       = env_name,
        backend        = "mjx",
        healthy_reward = 2.0,
    )
    eval_env = env
    env = envs.training.wrap(
        env,
        episode_length   = 1000,
        action_repeat    = 1,
        randomization_fn = None,
    )
    reset_fn = jax.jit(env.reset)

    # Prepare keys
    key = jax.random.PRNGKey(rng)
    key, env_key, train_state_key, epoch_key = jax.random.split(key, 4)
    env_keys = jax.random.split(env_key, num_envs)
    env_states = reset_fn(env_keys)

    # Create networks
    actor_network  = ActorNetwork((32, 32, 32, 32), env.action_size, linen.swish)
    critic_network = CriticNetwork((256,256,256,256,256), linen.swish)

    # Use one train state for simplicity
    train_state = create_train_state(
        actor_network,
        critic_network,
        env.observation_size,
        learning_rate,
        train_state_key,
    )

    loss_fn = functools.partial(
        ppo_loss_fn,
        entropy_cost     = entropy_cost,
        discount         = discount,
        gae_lambda       = gae_lambda,
        clipping_epsilon = clipping_epsilon,
        vf_coefficient   = vf_coefficient
    )

    def train_step(carry, tmp):
        train_state, env_states, key = carry
        key, learn_key, rollout_key = jax.random.split(key, 3)

        def get_trajectories(carry,_):
            env_states, rollout_key = carry

            trajectories, next_states, rollout_key, = unroll(
                train_state,
                env_states,
                rollout_key,
                env,
                unroll_length
            )

            return (next_states, rollout_key), trajectories


        # First: Collect our desired amount of trajectories (batch_size * num_minibatches // num_envs)
        (next_states, _), batched_trajectories = jax.lax.scan(
            f      = get_trajectories, 
            xs     = None, 
            init   = (env_states, rollout_key), 
            length = (batch_size * num_minibatches) // num_envs
        )

        # NOTE: `batched_trajectories` is of shape ~ [(batch_size * num_minibatches) // num_envs, unroll_length, num_envs, data]

        # Swap "unroll_length" and "num_envs" dims around -> [(batch_size * num_minibatches) // num_envs, num_envs, unroll_length, data]
        batched_trajectories = jax.tree_util.tree_map(lambda x: jp.swapaxes(x, 1, 2), batched_trajectories)

        # Now squash together the first two dims -> [(batch_size * num_minibatches)], unroll_length, data]
        batched_trajectories = jax.tree_util.tree_map(lambda x: jp.reshape(x, (-1,) + x.shape[2:]), batched_trajectories)

        # After collecting all our data, update the normalisation statistics
        normaliser_params = normalise.update_running_mean_std(
            train_state.normaliser_params,
            batched_trajectories.observation
        )
        train_state = train_state.replace(normaliser_params=normaliser_params)

        train_state, _, _ = learn(
            train_state,
            batched_trajectories,
            num_minibatches,
            updates_per_batch,
            loss_fn,
            learn_key,
        )

        return (train_state, next_states, key), None


    # For every eval we compute a training epoch , each training epoch is made up of x steps, where x == `steps_per_epoch` 
    def training_epoch(
        train_state: CustomTrainState,
        env_states   : envs.State,
        key          : jax.Array
    ):

        result, _ = jax.lax.scan(
            f      = train_step,
            init   = (train_state, env_states, key),
            length = steps_per_epoch,
            xs     = None
        )

        train_state, env_states, key = result
        return train_state, env_states, key


    training_epoch = jax.jit(training_epoch)

    jit_end_time = time.time()
    print(f"Time taken to JIT {jit_end_time - jit_start_time}\n")
    train_start_time = time.time()


    environment_steps = 0


    for eval in range(num_evals):
        # After every training epoch, we evaluate the current policy across 128 seeds
        train_state, env_states, epoch_key = training_epoch(
            train_state,
            env_states,
            epoch_key
        )

        environment_steps += steps_per_epoch * steps_per_train_step

        average_reward = evaluate_policy(
            actor_state = train_state,
            env         = env,
            key         = key,
            max_steps   = 1000
        )

        eval_idx = eval + 1
        log_eval(
        eval_idx=eval_idx,
        num_evals=num_evals,
        environment_steps=environment_steps,
        training_steps=training_steps,
            average_reward=average_reward,
            train_start_time=train_start_time,
        )


    print(f"Final environment step count: {environment_steps}")

    visualise_policy(
        actor_state = train_state,
        env         = eval_env,
        key         = key,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Agent")
    parser.add_argument(
        "--rng",
        required = False,
        default  = 0,
        type     = int,
    )

    args = parser.parse_args()
    main(args.rng)
