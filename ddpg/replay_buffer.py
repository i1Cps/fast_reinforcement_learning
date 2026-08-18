import jax
from jax import numpy as jp
from jax import flatten_util
from typing import Callable, Tuple
from .custom_types import JaxReplayBuffer, Transition

def init_buffer(
    capacity         : int,
    dummy_transition : Transition
) -> Tuple[JaxReplayBuffer, Callable]:

    dummy_flatten, unflatten_fn = flatten_util.ravel_pytree(dummy_transition)
    data_shape   = len(dummy_flatten)
    buffer_shape = (capacity, data_shape)

    return JaxReplayBuffer(
        data     =jp.zeros(buffer_shape, dtype=jp.float32),
        capacity =jp.array(capacity, dtype=jp.int32),
        ptr      =jp.array(0, dtype=jp.int32),
    ), unflatten_fn


def insert_data(
    data       : Transition,
    buffer     : JaxReplayBuffer,
    flatten_fn : Callable
) -> JaxReplayBuffer:

    # Flatten data in preparation for insert
    flattened_input_data = flatten_fn(data)

    buffer_data       = buffer.data
    capacity          = buffer.capacity
    insert_position   = buffer.ptr
    input_data_length = flattened_input_data.shape[0]

    # If new data exceeds capacity, discard the oldest entries
    roll = jp.minimum(0, capacity - insert_position - input_data_length)

    # Similar to if statement with functions
    buffer_data = jax.lax.cond(
        roll < 0, 
        lambda d: jp.roll(d, roll, axis=0), 
        lambda d: d, buffer_data
    )

    # Remember roll is negative or 0
    insert_position = insert_position + roll

    # Update the buffer using dynamic slice. The .at[..].set(..) is slow
    buffer_data = jax.lax.dynamic_update_slice_in_dim(
        buffer_data, flattened_input_data, insert_position, axis=0
    )

    # Update the insert position (end of buffer position is allowed with + 1)
    insert_position = (insert_position + input_data_length) % (buffer_data.shape[0] + 1)

    return JaxReplayBuffer(
        data     = buffer_data,
        capacity = capacity,
        ptr      = insert_position,
    )


def sample_buffer(
    buffer       : JaxReplayBuffer,
    batch_size   : int,
    key          : jax.Array,
    unflatten_fn : Callable
) -> Transition:

    key, subkey  = jax.random.split(key)
    indices      = jax.random.randint(subkey, (batch_size,), 0, buffer.ptr)
    sampled_data = jp.take(buffer.data, indices, axis=0)
    return unflatten_fn(sampled_data)
