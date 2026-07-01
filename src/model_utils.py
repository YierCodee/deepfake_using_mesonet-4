import tensorflow as tf
import numpy as np
import streamlit as st
from .config import IMG_SIZE, MODEL_PATH, INVERT_PROBABILITY


@st.cache_resource
def load_model(model_path=None):
    """Load the MesoNet model once and cache it for the session."""
    if model_path is None:
        model_path = MODEL_PATH
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        # Dry run to build the model graph (needed for model.input to exist)
        dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
        model(dummy, training=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None


def _maybe_invert(raw):
    """Invert probability if model was trained with swapped labels."""
    return 1.0 - raw if INVERT_PROBABILITY else raw


def predict(model, img_array):
    """Predict a single preprocessed image. Returns float probability of FAKE."""
    raw = float(model.predict(img_array, verbose=0)[0][0])
    return _maybe_invert(raw)


def predict_batch(model, batch_array):
    """Predict a batch of preprocessed images. Returns array of FAKE probabilities."""
    raw = model.predict(batch_array, verbose=0)[:, 0]
    return _maybe_invert(raw)


def find_last_conv_layer(model):
    """Return the name of the last Conv2D layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def compute_gradcam(model, img_array):
    """
    Compute a Grad-CAM heatmap for a single prediction.

    Walks through each layer inside a GradientTape so that the tape
    can track the dependency chain from the target conv layer to the
    final loss.

    Returns a 2D numpy array (height x width) in [0, 1],
    or None if no convolutional layer is found.
    """
    layer_name = find_last_conv_layer(model)
    if layer_name is None:
        return None

    x = tf.constant(img_array)
    target_activation = None

    with tf.GradientTape() as tape:
        for layer in model.layers:
            x = layer(x, training=False)
            if layer.name == layer_name:
                target_activation = x

        loss = x[:, 0]

    if target_activation is None:
        return None

    grads = tape.gradient(loss, target_activation)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(
        tf.multiply(pooled_grads, target_activation[0]), axis=-1
    )
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())

    return heatmap.numpy()
