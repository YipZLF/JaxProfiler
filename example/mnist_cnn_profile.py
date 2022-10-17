import jax
import jax.numpy as jnp  # JAX NumPy

from flax import linen as nn  # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

import numpy as np  # Ordinary NumPy
import optax  # Optimizers

# import tensorflow_datasets as tfds  # TFDS for MNIST

from functools import partial 

import datasets
import timer
t = timer.get_timer()

# Network
class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

def cross_entropy_loss(*, logits, labels):
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()


def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics


def create_train_state(rng, learning_rate, momentum):
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
    """Train for a single step."""

    def loss_fn(params):
        logits = CNN().apply({'params': params}, batch['image'])
        loss = cross_entropy_loss(logits=logits, labels=batch['label'])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=batch['label'])
    return state, metrics


@jax.jit
def eval_step(params, batch):
    logits = CNN().apply({'params': params}, batch['image'])
    return compute_metrics(logits=logits, labels=batch['label'])


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    train_ds = {}
    test_ds = {}
    train_ds['image'], train_ds['label'], test_ds['image'], test_ds['label'] = datasets.mnist_raw()
    train_ds['image'], train_ds['label'], test_ds['image'], test_ds['label'] = train_ds['image'].reshape(
        -1, 28, 28, 1) / np.float32(255.), train_ds['label'], test_ds['image'].reshape(
            -1, 28, 28, 1) / np.float32(255.), test_ds['label']
    return train_ds, test_ds


def train_epoch(state, train_ds, batch_size, epoch, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    # t("train step").start()
    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, metrics = train_step(state, batch)
        batch_metrics.append(metrics)
    # t("train step").stop()

    # t.log()
    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}

    print('train epoch: %d, loss: %.4f, accuracy: %.2f' %
          (epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

    return state


def eval_model(params, test_ds):
    metrics = eval_step(params, test_ds)
    metrics = jax.device_get(metrics)
    summary = jax.tree_util.tree_map(lambda x: x.item(), metrics)
    return summary['loss'], summary['accuracy']


if __name__ == "__main__":
    train_ds, test_ds = get_datasets()
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    learning_rate = 0.1
    momentum = 0.9

    state = create_train_state(init_rng, learning_rate, momentum)
    del init_rng  # Must not be used anymore.

    # print(jax.make_jaxpr(train_step)(state,{k: v[:128, ...] for k, v in train_ds.items()} ))
    # print(jax.make_jaxpr(eval_step)(state.params, test_ds ))
    # exit()
    num_epochs = 10
    batch_size = 128

    # with jax.profiler.trace("./jax_trace",create_perfetto_link=False):


    # rng, input_rng = jax.random.split(rng)
    for epoch in range(1, num_epochs + 1):
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        with jax.profiler.StepTraceAnnotation("TRAINING STEP", step_num=epoch):
            state = train_epoch(state, train_ds, batch_size, epoch, input_rng)
        # Evaluate on the test set after each training epoch
        with jax.profiler.StepTraceAnnotation("EVALUATION STEP", step_num=epoch):
            test_loss, test_accuracy = eval_model(state.params, test_ds)
        print(' test epoch: %d, loss: %.2f, accuracy: %.2f' % (epoch, test_loss, test_accuracy * 100))
    
    # key = jax.random.PRNGKey(0)
    # x = jax.random.normal(key, (1,1))
    # x.block_until_ready()

    # jax.profiler.stop_trace()
