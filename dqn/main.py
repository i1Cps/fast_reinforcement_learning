import argparse
import os
import time

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

import jax
from jax import numpy as jp
from jax import flatten_util
import gymnax
from gymnax import EnvState
from .network import QNetwork
from .replay_buffer import init_buffer, insert_data, sample_buffer
from .policy import create_train_state, learn, update_target_params
from .actor import evaluate_policy, env_step, visualise_policy
from .custom_types import CustomTrainState, JaxReplayBuffer, Transition
from .logger import log_eval


def main(rng: int):
    # NOTE: DQN hyperparameters
    training_steps         = 1_000_000
    num_envs               = 256
    batch_size             = 256
    learning_rate          = 0.00012
    updates_per_env_step   = 4
    replay_buffer_capacity = 200_000
    epsilon                = 0.9
    epsilon_decay_rate     = 0.998
    discount               = 0.99
    target_update_interval = 100
    num_evals              = 10
    env_name               = 'CartPole-v1'
    steps_per_eval = training_steps // num_evals
    steps_per_epoch = steps_per_eval // num_envs

    print(f"\n\nWelcome to jax rl, the functional way! \n\n \
        We're about to train our agent on the environment {env_name} using DQN\n")

    print(f"""Training Params: \n
        Training steps                        : {training_steps}
        Number of parallel environments       : {num_envs}
        Batch size for each gradient update   : {batch_size}
        Number of gradient updates per step   : {updates_per_env_step}
        Number of steps per epoch             : {steps_per_epoch}
        Total number of gradient updates      : {updates_per_env_step * steps_per_epoch * num_evals}
    """)

    # Measure compilation time
    jit_start_time = time.time()

    # Prepare the parallel environments
    env, env_params = gymnax.make(env_name)
    eval_env = env

    # Prepare environment functions with JIT
    reset_fn = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    step_env_fn = jax.jit(jax.vmap(env.step, in_axes=(None, 0, 0)))
    get_obs_fn = jax.jit(jax.vmap(env.get_obs, in_axes=(0, None, None)))

    # Prepare keys
    key = jax.random.PRNGKey(rng)
    env_key, policy_key, key = jax.random.split(key, 3)
    env_keys = jax.random.split(env_key, num_envs)
    obs, env_states = reset_fn(env_keys, None)


    # Create network
    network = QNetwork((64,64), env.num_actions)

    train_state = create_train_state(
        network,
        env.obs_shape[0],
        learning_rate,
        policy_key
    )

    # Create the replay buffer
    dummy_obs        = jp.zeros(env.obs_shape)
    dummy_actions    = jp.zeros(())
    dummy_transition = Transition(
        observation      = dummy_obs,
        action           = dummy_actions,
        reward           = jp.array(0.0, dtype=jp.float32),
        done             = jp.array(0.0, dtype=jp.float32),
        next_observation = dummy_obs,
    )

    buffer, unflatten_fn = init_buffer(replay_buffer_capacity, dummy_transition)

    # Keep flatten and unflatten functions, then vmap them so it auto flattens a whole batch of the Transition dataclass
    flatten_fn = jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])
    unflatten_fn = jax.vmap(unflatten_fn)

    # Learning epoch
    def train_step(carry, tmp):
        train_state, env_states, buffer, epsilon, key, learning_steps = carry


        # Split keys
        next_key, rollout_key, buffer_key = jax.random.split(key, 3)

         # Generate and insert data into the buffer
        final_env_states, data = env_step(
            train_state = train_state,
            env_states  = env_states,
            epsilon     = epsilon,
            key         = rollout_key,
            get_obs     = get_obs_fn,
            step_env    = step_env_fn
        )
        epsilon = jp.maximum(epsilon * epsilon_decay_rate, 0.1)

        buffer = insert_data(data, buffer, flatten_fn)

        # Sample data from buffer
        data = sample_buffer(buffer, batch_size * updates_per_env_step, buffer_key, unflatten_fn)

        # Reshape data for scan function:
        # (updates_per_step * batch_size, ...) - > (update_per_step, batch_size, ...)
        data = jax.tree_util.tree_map(
            lambda x: jp.reshape(x, (updates_per_env_step, -1) + x.shape[1:]),
            data,
        )

        # Track the number of gradient updates for target network updates
        step_indices = learning_steps + jp.arange(updates_per_env_step)

        # Apply one gradient update
        def one_learn_step(carry, xs):

            train_state = carry
            data, i = xs # unpack (data, step number)

            # Learn then update target params of both actor and critic
            train_state = learn(train_state, discount, data)

            # Update target networks at the selected interval
            update_targets = ((i + 1) % target_update_interval) == 0

            def _update(train_state):
                return update_target_params(train_state)

            def _skip(train_state):
                return train_state

            train_state = jax.lax.cond(
                update_targets, 
                _update,
                _skip,
                operand= train_state
            )
            return train_state, None

        result, _ = jax.lax.scan(
            f     = one_learn_step,
            init  = train_state,
            xs    = (data, step_indices)
        )
        train_state = result

        return (
            train_state,
            final_env_states,
            buffer,
            epsilon,
            next_key,
            learning_steps + updates_per_env_step,
        ), None

    # Every Eval we compute a training epoch for `steps_per_epoch` number of steps
    def training_epoch(
        train_state : CustomTrainState,
        env_states  : EnvState,
        buffer      : JaxReplayBuffer,
        epsilon     : float,
        key         : jax.Array,
        learning_steps : int,
    ):

        result, _  = jax.lax.scan(
            f      = train_step,
            init   = (
                train_state,
                env_states,
                buffer,
                epsilon,
                key,
                learning_steps,
            ),
            length = steps_per_epoch,
            xs     = None
        )

        train_state, env_states, buffer, epsilon, key, learning_steps = result
        return train_state, env_states, buffer, epsilon, key, learning_steps

    training_epoch = jax.jit(training_epoch)

    jit_end_time = time.time()
    print(f"Time taken to JIT: {jit_end_time - jit_start_time}\n")

    epoch_key, key = jax.random.split(key, 2)

    learning_steps = 0
    environment_steps = 0

    train_start_time = time.time()
    for eval in range(num_evals):
        # After every training epoch, we evaluate the current policy across 128 seeds
        train_state, env_states, buffer, epsilon, epoch_key, learning_steps = training_epoch(
            train_state,
            env_states,
            buffer,
            epsilon,
            epoch_key,
            learning_steps,
        )
        # jax.debug.print("epsilon: {}",epsilon)

        environment_steps += steps_per_epoch * num_envs

        average_reward = evaluate_policy(
            train_state = train_state,
            key         = key,
            reset       = reset_fn,
            get_obs     = get_obs_fn,
            step_env    = step_env_fn,
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

    print(f"Final learning step count: {learning_steps}")

    visualise_policy(
        train_state=train_state,
        env_name=env_name,
        rollout_length=500,
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
