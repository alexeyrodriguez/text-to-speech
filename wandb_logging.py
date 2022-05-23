import wandb
import tensorflow as tf

## Logging functions borrowed from Wandb

def log_gradients(model, gradients):
    metrics = {}
    weights = model.trainable_weights
    for (weight, gradients) in zip(weights, gradients):
        metrics[
            "gradients/" + weight.name.split(":")[0] + ".gradient"
        ] = wandb.Histogram(tf.convert_to_tensor(gradients))
    return metrics

def log_weights(model):
    metrics = {}
    weights = model.trainable_weights
    for weight in weights:
        metrics[
            "parameters/" + weight.name.split(":")[0] + ".weights"
        ] = wandb.Histogram(tf.convert_to_tensor(weight))
    return metrics


def _array_has_dtype(array):
    return hasattr(array, "dtype")


def _update_if_numeric(metrics, key, values):
    if not _array_has_dtype(values):
        _warn_not_logging(key)
        return

    if not is_numeric_array(values):
        _warn_not_logging_non_numeric(key)
        return

    metrics[key] = wandb.Histogram(values)


def is_numeric_array(array):
    return np.issubdtype(array.dtype, np.number)


def _warn_not_logging_non_numeric(name):
    wandb.termwarn(
        "Non-numeric values found in layer: {}, not logging this layer".format(name),
        repeat=False,
    )


def _warn_not_logging(name):
    wandb.termwarn(
        "Layer {} has undetermined datatype not logging this layer".format(name),
        repeat=False,
    )


def log(epoch, model, metrics, gradients=None):
    wandb.log(log_weights(model), commit=False)

    if gradients:
        wandb.log(log_gradients(model, gradients), commit=False)

    wandb.log({'epoch': epoch}, commit=False)
    wandb.log(metrics, commit=True)
