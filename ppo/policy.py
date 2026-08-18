import jax
import optax
from jax import numpy as jp

from .custom_types import CustomTrainState
from .normalise import init_running_mean_std, normalize_obs
import distrax

# This creates a train state containing network parameters, optimizer state, and the network apply function
def create_train_state(
    actor_network,
    critic_network, 
    input_dims, 
    learning_rate,
    key
):
    actor_key, critic_key = jax.random.split(key, 2)

    dummy_obs     = jp.zeros((1, input_dims))
    actor_variables     = actor_network.init(actor_key, dummy_obs)
    critic_variables    = critic_network.init(critic_key, dummy_obs)

    # We will model the params object as a dict containing both the actor and critic params, 
    # and use their respective apply function manually
    params = {
        "actor"  : actor_variables,
        "critic" : critic_variables,
        "log_std" : jp.zeros((actor_network.n_actions,)),
    }

    ppo_train_state = CustomTrainState.create(
        apply_fn  = {"actor": actor_network.apply, "critic": critic_network.apply},
        params    = params,
        tx        = optax.adam(learning_rate),
        normaliser_params = init_running_mean_std(dummy_obs.shape[1:]),
    )

    return ppo_train_state

def choose_actions(
    train_state: CustomTrainState,
    observations: jp.ndarray,
    key: jax.Array,
    deterministic: bool = False,
):
    # Normalize observations before the policy forward pass using running stats.
    obs = normalize_obs(observations, train_state.normaliser_params)
    mean = train_state.apply_fn["actor"](train_state.params["actor"], obs)
    std = jp.broadcast_to(jp.exp(train_state.params["log_std"]), mean.shape)
    dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)

    # Sample the actions from the distribution, sample in pre-tanh space to avoid atanh instability
    raw_actions = dist.sample(seed=key) if not deterministic else mean 

    # Calculate the log probabilities in raw space then use a jacobian correction to transfer them to tanh space
    log_probs_raw = dist.log_prob(raw_actions)
    log_det = 2.0 * (jp.log(2.0) - raw_actions - jax.nn.softplus(-2.0 * raw_actions))
    log_prob = log_probs_raw - jp.sum(log_det, axis=-1)

    # Postprocess into env actions in [-1, 1].
    actions = jp.tanh(raw_actions)
    return actions, {"log_probs": log_prob, "raw_actions": raw_actions}


def learn(
    train_state       : CustomTrainState,
    trajectories,
    num_minibatches   : int,
    updates_per_batch : int,
    loss_fn,
    rng               : jax.Array
):

    def minibatch_step(carry, mini_batch):
        # Take a minibatch of trajectories and send it off to the loss function
        train_state, minibatch_rng = carry

        minibatch_rng, loss_rng = jax.random.split(minibatch_rng)

        loss, gradients = jax.value_and_grad(loss_fn, argnums=0)(
            train_state.params,
            mini_batch,
            train_state,
            rng = loss_rng
        )

        # Update the network weights
        updated_train_state = train_state.apply_gradients(grads=gradients)
        return (updated_train_state, minibatch_rng), loss

    # Learning epoch
    def run_epoch(carry, _):
        train_state, epoch_rng = carry

        epoch_rng, shuffle_rng, minibatch_rng= jax.random.split(epoch_rng, 3)

        # Shuffle and batch trajectories into mini batches
        def shuffle_and_batch_data(x: jp.ndarray):
            x = jax.random.permutation(shuffle_rng, x)
            x = jp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x
        minibatches = jax.tree_util.tree_map(shuffle_and_batch_data, trajectories)

        (new_train_state, _), minibatch_losses = jax.lax.scan(
            f      = minibatch_step,
            xs     = minibatches,
            init   = (train_state, minibatch_rng),
            length = num_minibatches,
        )

        return (new_train_state, epoch_rng), minibatch_losses

    # Perform a learning epoch x times on this batch of data
    (train_state, rng), epoch_losses = jax.lax.scan(
        f      = run_epoch,
        xs     = None,
        init   = (train_state, rng),
        length = updates_per_batch
    )

    return train_state, rng, epoch_losses
