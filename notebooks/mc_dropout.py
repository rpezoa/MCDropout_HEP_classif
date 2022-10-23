import numpy as np
import tensorflow as tf


def proba_to_class(proba, threshold=0.5):
    if proba.shape[-1] == 1:
        return np.where(proba > threshold, 1, 0)
    else:
        return proba.argmax(axis=1)


def get_mc_model(exported_automodel, metrics, p=0.25, loss='binary_crossentropy'):
    """Add MC dropout to a autokeras optimized model architecture. It returns
    a new keras model with extra dropout layers after each dense layer.
    
    Args:
        exported_automodel (autokeras....): A optimized architecture.
        metrics (list(tf.keras.Metrics)): List of metrics.
        p (float): The dropout rate for MC-dropout.
    
    Returns:
        The new keras model with MC-dropout. This model isn't fit.
    """
    input_shape = exported_automodel.layers[0].input_shape[0][1]
    f = input_layer = tf.keras.Input(shape=(input_shape), name="input")
    for layer in exported_automodel.layers[1:]:
        if isinstance(layer, tf.keras.layers.Dropout):
            continue 
        f = type(layer)(**layer.get_config())(f)
        if isinstance(layer, (tf.keras.layers.ReLU,)):
            f = tf.keras.layers.Dropout(rate=p)(f, training=True)
    mc_model = tf.keras.Model(input_layer, f, name="mc_model")
    # Optimizer
    try:
        optimizer_parameters = dict(exported_automodel.optimizer.get_config())
        optimizer_name = optimizer_parameters['name']
        del optimizer_parameters['name']
        mc_model_opt = tf.keras.optimizers.get({'class_name': optimizer_name, 'config': optimizer_parameters})
    except:
        print('Using default optimizer')
        mc_model_opt = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1 = 0.9,
            beta_2 = 0.999,
            epsilon = 1e-06,
            amsgrad = False,
        )
    # Loss
    mc_model.compile(optimizer=mc_model_opt, loss=loss, metrics=metrics)
    return mc_model


def predictive_distribution(mc_predictions):
    """Compute the predictive distributions given T forward pass samples for a 
    batch of inputs, or a single instance.
    
    Args:
        mc_predictions (np.ndarray): Monte-Carlo predictions with shape (T, batch_size, 1) for a batch
            or (T, 1) for a single instance
    """
    predictive_distributions = mc_predictions.mean(axis=0)
    if mc_predictions.ndim == 2:
        return predictive_distributions[0]
    return predictive_distributions


def _to_batch_format(y_predictive_distributions, is_sample):
    if is_sample:
        if y_predictive_distributions.ndim == 0:
            y_predictive_distributions = np.expand_dims(y_predictive_distributions, (0, 1))
        else:
            y_predictive_distributions = np.expand_dims(y_predictive_distributions, 0)
    n_classes = y_predictive_distributions.shape[-1]
    if n_classes == 1:
        y_predictive_distributions = np.hstack((y_predictive_distributions, 1 - y_predictive_distributions))
        n_classes = 2
    return y_predictive_distributions, n_classes


def predictive_entropy(y_predictive_distributions, is_sample=False, normalize=False):
    """Compute the predictive entropy given T forward pass samples for a batch
    of inputs.
    
    Predictive Entropy is the average amount of information contained in 
    the predictive distribution. Predictive entropy is a biases estimator.
    The bias of this estimator will decrease as $T$ (`sample_size`) increases.
    
    https://arxiv.org/pdf/1803.08533.pdf
    
    Args:
        y_predictive_distribution (np.ndarray|np.float): Model's predictive distributions. 
            With shape (batch_size, classes), or  a np.float if is a sample.
        is_sample: is is just one sample:
        normalize:  `bool`
            Change range into [0,1]
    Returns:
        `np.ndarray` with shape (batch_size,).
            Return predictive entropy for a batch. Or a single value if is a sample
    """
    # To batch format
    y_predictive_distributions, n_classes = _to_batch_format(y_predictive_distributions, is_sample)
    # Numerical Stability
    eps = np.finfo(y_predictive_distributions.dtype).tiny #.eps
    y_log_predictive_distributions = np.log(eps + y_predictive_distributions)
    # Predictive Entropy
    H = -1*np.sum(y_predictive_distributions * y_log_predictive_distributions, axis=1)
    if normalize:
        H = H/np.log(n_classes)
    return H[0] if is_sample else H


def mutual_information(y_predictive_distributions, y_predictions_samples, is_sample=False, normalize=False):
    """Compute the mutual information given T forward pass samples for a batch
    of inputs.
    
    Measure of the mutual dependence between the two variables. More 
    specifically, it quantifies the "amount of information" (in units
    such as shannons, commonly called bits) obtained about one random
    variable through observing the other random variable.
    
    https://arxiv.org/pdf/1803.08533.pdf
    
    Args:
        y_predictive_distribution (np.ndarray|np.float):  Model's predictive distributions.
            With shape (batch_size, classes), or  a np.float if is a sample
        y_predictions_samples (`np.ndarray`):  Forward pass samples, with shape
            (sample size, batch_size, classes) for a batch, or (sample size, classes) for a sample.
        is_sample: is is just one sample:
        normalize:  `bool`
            Change range into [0,1]
    Returns:
        `np.ndarray` with shape (batch_size,).
            Return predictive entropy for a batch. Or a single value if is a sample
    """
    # To batch format
    y_predictive_distributions, n_classes = _to_batch_format(y_predictive_distributions, is_sample)
    y_predictions_samples = np.array([_to_batch_format(s, is_sample)[0] for s in y_predictions_samples])
    # Numerical Stability 
    eps = np.finfo(y_predictive_distributions.dtype).tiny #.eps
    ## Entropy (batch, classes)
    y_log_predictive_distribution = np.log(eps + y_predictive_distributions) 
    H = -1*np.sum(y_predictive_distributions * y_log_predictive_distribution, axis=1)
    ## Expected value (batch, classes) 
    sample_size = len(y_predictions_samples)
    y_predictions_samples = np.swapaxes(y_predictions_samples, 0, 1)
    y_log_predictions_samples = np.log(eps + y_predictions_samples)
    minus_E = np.sum(y_predictions_samples*y_log_predictions_samples, axis=(1,2))
    minus_E /= sample_size
    ## Mutual Information
    I = H + minus_E
    if normalize:
        I = I/np.log(n_classes)
    return I[0] if is_sample else I
