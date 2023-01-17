from typing import Optional, Tuple

import math
import tensorflow as tf
from tensorflow.keras import layers

from utils import shape_list, stable_softmax, get_initializer
from configuration_wav2vec2 import config
from custom_ops import wav2vec2_weight_norm_conv1d, wav2vec2_group_norm, wav2vec2_attention


def gelu(x: tf.Tensor):
    x = tf.convert_to_tensor(x)
    pi = tf.cast(math.pi, x.dtype)
    coeff = tf.cast(0.044715, x.dtype)
    cdf = 0.5 * (1.0 + tf.tanh(tf.sqrt(2.0 / pi) * (x + coeff * tf.pow(x, 3))))

    return x * cdf


def wav2vec2_groupnorm_conv_layer(layer_id: int):
    def _wav2vec2_groupnorm_conv_layer(hidden_states: tf.Tensor):
        in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        out_conv_dim = config.conv_dim[layer_id]
        hidden_states = layers.Conv1D(
            filters=out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
        )(hidden_states)
        # hidden_states = TFWav2Vec2GroupNorm(
        #     groups=out_conv_dim, epsilon=config.layer_norm_eps)(hidden_states)

        hidden_states = wav2vec2_group_norm(
            groups=out_conv_dim, epsilon=config.layer_norm_eps)(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states
    return _wav2vec2_groupnorm_conv_layer


def wav2vec2_no_layernorm_conv_layer(layer_id: int):
    def _wav2vec2_no_layernorm_conv_layer(hidden_states: tf.Tensor):
        in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        out_conv_dim = config.conv_dim[layer_id]
        hidden_states = layers.Conv1D(
            filters=out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
        )(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states
    return _wav2vec2_no_layernorm_conv_layer


def wav2vec2_positional_conv_embedding(hidden_states: tf.Tensor):
    hidden_states = wav2vec2_weight_norm_conv1d(
        filters=config.hidden_size,
        kernel_size=config.num_conv_pos_embeddings,
        groups=config.num_conv_pos_embedding_groups,
        explicit_padding=config.num_conv_pos_embeddings // 2,
    )(hidden_states)

    num_pad_remove = 1 if config.num_conv_pos_embeddings % 2 == 0 else 0
    if num_pad_remove > 0:
        hidden_states = hidden_states[:, : -num_pad_remove, :]
    hidden_states = gelu(hidden_states)
    return hidden_states


def wav2vec2_feed_forward(hidden_states: tf.Tensor, training: bool = False):
    intermediate_dropout = tf.keras.layers.Dropout(
        config.activation_dropout)

    intermediate_dense = tf.keras.layers.Dense(
        units=config.intermediate_size,
        kernel_initializer=get_initializer(config.initializer_range),
        bias_initializer="zeros",
    )

    output_dense = tf.keras.layers.Dense(
        units=config.hidden_size,
        kernel_initializer=get_initializer(config.initializer_range),
        bias_initializer="zeros",
    )
    output_dropout = tf.keras.layers.Dropout(config.hidden_dropout)

    hidden_states = intermediate_dense(hidden_states)
    hidden_states = gelu(hidden_states)
    hidden_states = intermediate_dropout(hidden_states, training=training)

    hidden_states = output_dense(hidden_states)
    hidden_states = output_dropout(hidden_states, training=training)
    return hidden_states


def wav2vec2_encoder_layer(hidden_states: tf.Tensor,
                           attention_mask: Optional[tf.Tensor] = None,
                           output_attentions: Optional[bool] = False,
                           training: bool = False,
                           name: str = None):
    attn_residual = hidden_states
    # hidden_states, attn_weights, _ = TFWav2Vec2Attention(
    #     embed_dim=config.hidden_size,
    #     num_heads=config.num_attention_heads,
    #     dropout=config.attention_dropout,
    #     is_decoder=False,
    # )(
    #     hidden_states, attention_mask=attention_mask, training=training
    # )
    hidden_states, attn_weights, _ = wav2vec2_attention(
        embed_dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        dropout=config.attention_dropout,
        is_decoder=False,
    )(
        hidden_states, attention_mask=attention_mask, training=training
    )
    hidden_states = layers.Dropout(config.hidden_dropout)(
        hidden_states, training=training)
    hidden_states = attn_residual + hidden_states

    hidden_states = layers.LayerNormalization(
        epsilon=config.layer_norm_eps)(hidden_states)
    hidden_states = hidden_states + \
        wav2vec2_feed_forward(hidden_states, training=False)
    hidden_states = layers.LayerNormalization(
        epsilon=config.layer_norm_eps
    )(hidden_states)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs
