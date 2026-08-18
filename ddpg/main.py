import argparse
import os
import time

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

import jax
from jax import flatten_util
from jax import numpy as jp
from brax import envs
from flax import linen
from .replay_buffer import init_buffer, insert_data, sample_buffer
from .network import ActorNetwork, CriticNetwork
from .policy import create_train_states, learn, update_target_params
from .actor import evaluate_policy, visualise_policy, env_step
from .custom_types import CustomTrainState, JaxReplayBuffer, Transition
from .logger import log_eval

def main(rng: int):
    # NOTE: DDPG hyperparameters
    training_steps          = 20_000_000
    num_envs                = 2048
    batch_size              = 256
    updates_per_env_step    = 32
    replay_buffer_capacity  = 5_000_000
    actor_learning_rate     = 0.0003
    critic_learning_rate    = 0.0003
    discount                = 0.990
    tau                     = 0.005
    target_update_interval  = 2
    num_evals               = 10
    env_name                = 'halfcheetah'
    steps_per_eval = training_steps // num_evals
    steps_per_epoch = steps_per_eval // num_envs

    print(f"Welcome to jax rl, the functional way! \n\n \
    We're about to train our agent on the environment {env_name} using DDPG\n")

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
    env = envs.get_environment(env_name=env_name, backend="mjx")
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
    env_key, policy_key, key = jax.random.split(key, 3)
    env_keys = jax.random.split(env_key, num_envs)
    env_states = reset_fn(env_keys)

    # Create networks
    actor_network = ActorNetwork((256,256), env.action_size, linen.swish)
    critic_network = CriticNetwork((256,256), linen.swish)

    actor_state, critic_state = create_train_states(
        actor_network,
        critic_network,
        env.observation_size,
        env.action_size,
        critic_learning_rate,
        actor_learning_rate,
        policy_key,
    )

    # Create the replay buffer
    dummy_obs        = jp.zeros((env.observation_size))
    dummy_actions    = jp.zeros((env.action_size))
    dummy_transition = Transition(
        observation      = dummy_obs,
        action           = dummy_actions,
        reward           = jp.array(0.0, dtype=jp.float32),
        done             = jp.array(0.0, dtype=jp.float32),
        truncation       = jp.array(0.0, dtype=jp.float32),
        next_observation = dummy_obs,
    )

    buffer, unflatten_fn = init_buffer(replay_buffer_capacity, dummy_transition)

    # Keep flatten and unflatten functions, then vmap them so it auto flattens a whole batch of the Transition dataclass
    flatten_fn = jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])
    unflatten_fn = jax.vmap(unflatten_fn)

    # Generate and insert data into buffer using random actions
    def warmup(
        actor_state : CustomTrainState,
        buffer      : JaxReplayBuffer,
        env_states  : envs.State,
        key         : jax.Array
    ):
        def f (carry, tmp):
            env_states, buffer, key = carry
            key, subkey = jax.random.split(key)
            final_env_states, data = env_step(actor_state, env_states, True, subkey, env)
            buffer = insert_data(data, buffer, flatten_fn)
            return (final_env_states, buffer, key), None

        result, _ = jax.lax.scan(
            f      = f,
            init   = (env_states, buffer, key),
            xs     = None, 
            length = 100
        )
        final_env_states, _, _ = result 
        return final_env_states, buffer


    # Learning epoch
    def train_step(carry, tmp):
        actor_state, critic_state, env_states, buffer, key, learning_steps = carry

        # Split keys
        next_key, rollout_key, buffer_key = jax.random.split(key, 3)

         # Generate and insert data into the buffer
        final_env_states, data = env_step(actor_state, env_states, False, rollout_key, env)
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

            data, i = xs # unpack (data, step number)

            # Learn then update target params of both actor and critic
            actor_state, critic_state = carry
            actor_state, critic_state = learn(actor_state, critic_state, discount, data)

            # Update target networks at the selected interval
            update_targets = ((i + 1) % target_update_interval) == 0

            def _update(states):
                actor_state, critic_state = states
                return update_target_params(actor_state, critic_state, tau)

            def _skip(states):
                actor_state, critic_state = states
                return actor_state, critic_state

            actor_state, critic_state = jax.lax.cond(
                update_targets, 
                _update,
                _skip,
                operand=(actor_state, critic_state)
            )
            return (actor_state, critic_state), None

        result, _ = jax.lax.scan(
            f     = one_learn_step,
            init  = (actor_state, critic_state),
            xs    = (data, step_indices)
        )
        actor_state, critic_state = result

        return (
            actor_state,
            critic_state,
            final_env_states,
            buffer,
            next_key,
            learning_steps + updates_per_env_step,
        ), None


    # Every Eval we compute a training epoch for `steps_per_epoch` number of steps
    def training_epoch(
        actor_state  : CustomTrainState,
        critic_state : CustomTrainState,
        env_states   : envs.State,
        buffer       : JaxReplayBuffer,
        key          : jax.Array,
        learning_steps : int,
    ):

        result, _  = jax.lax.scan(
            f      = train_step,
            init   = (
                actor_state,
                critic_state,
                env_states,
                buffer,
                key,
                learning_steps,
            ),
            length = steps_per_epoch,
            xs     = None
        )

        actor_state, critic_state, env_states, buffer, key, learning_steps = result
        return actor_state, critic_state, env_states, buffer, key, learning_steps

    training_epoch = jax.jit(training_epoch)
    warmup         = jax.jit(warmup, donate_argnums=(1,2))

    jit_end_time = time.time()
    print(f"Time taken to JIT {jit_end_time - jit_start_time}\n")

    warmup_key, epoch_key, key = jax.random.split(key, 3)

    env_states, buffer = warmup(actor_state, buffer, env_states, warmup_key)

    learning_steps = 0
    environment_steps = 0

    train_start_time = time.time()
    for eval in range(num_evals):
        # After every training epoch, we evaluate the current policy across 128 seeds
        actor_state, critic_state, env_states, buffer, epoch_key, learning_steps = training_epoch(
            actor_state, 
            critic_state,
            env_states,
            buffer,
            epoch_key,
            learning_steps,
        )

        environment_steps += steps_per_epoch * num_envs

        average_reward = evaluate_policy(
            actor_state = actor_state,
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


    print(f"Final learning step count: {learning_steps}")

    visualise_policy(
        actor_state = actor_state,
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
