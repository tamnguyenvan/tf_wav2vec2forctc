from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from utils import shape_list, get_initializer
from configuration_wav2vec2 import config
from ops import (
    wav2vec2_groupnorm_conv_layer, wav2vec2_no_layernorm_conv_layer,
    wav2vec2_positional_conv_embedding, wav2vec2_encoder_layer
)

import tensorflow_model_optimization as tfmot

quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer


LARGE_NEGATIVE = -1e8


def _sample_without_replacement(distribution, num_samples):
    """
    Categorical sampling without replacement is currently not implemented. The gumbel-max trick will do for now - see
    https://github.com/tensorflow/tensorflow/issues/9260 for more info
    """
    z = -tf.math.log(tf.random.uniform(shape_list(distribution), 0, 1))
    _, indices = tf.nn.top_k(distribution + z, num_samples)
    return indices


def _scatter_values_on_batch_indices(values, batch_indices, output_shape):
    """
    Scatter function as in PyTorch with indices in format (batch_dim, indixes)
    """
    indices_shape = shape_list(batch_indices)
    # broadcast batch dim to indices_shape
    broad_casted_batch_dims = tf.reshape(
        tf.broadcast_to(tf.expand_dims(
            tf.range(indices_shape[0]), axis=-1), indices_shape), [1, -1]
    )
    # transform batch_indices to pair_indices
    pair_indices = tf.transpose(
        tf.concat([broad_casted_batch_dims, tf.reshape(batch_indices, [1, -1])], 0))
    # scatter values to pair indices
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), output_shape)


def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    min_masks: int = 0,
) -> tf.Tensor:
    """
    Computes random mask spans for a given shape
    Args:
        shape: the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        attention_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob:
            probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_length: size of the mask
        min_masks: minimum number of masked spans
    Adapted from [fairseq's
    data_utils.py](https://github.com/pytorch/fairseq/blob/e0788f7007a8473a76db573985031f3c94201e79/fairseq/data/data_utils.py#L376).
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and"
            f" `sequence_length`: {sequence_length}`"
        )
    # compute number of masked spans in batch
    num_masked_spans = mask_prob * sequence_length / \
        mask_length + tf.random.uniform((1,))
    num_masked_spans = tf.maximum(num_masked_spans, min_masks)
    num_masked_spans = tf.cast(num_masked_spans, tf.int32)

    # make sure num masked indices <= sequence_length
    num_masked_spans = tf.math.minimum(
        sequence_length // mask_length, num_masked_spans)
    num_masked_spans = tf.squeeze(num_masked_spans)

    # SpecAugment mask to fill
    spec_aug_mask = tf.zeros((batch_size, sequence_length), dtype=tf.int32)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = tf.ones((batch_size, sequence_length - (mask_length - 1)))

    # get random indices to mask
    spec_aug_mask_idxs = _sample_without_replacement(
        uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = tf.expand_dims(spec_aug_mask_idxs, -1)
    spec_aug_mask_idxs = tf.tile(spec_aug_mask_idxs, (1, 1, mask_length))
    spec_aug_mask_idxs = tf.reshape(
        spec_aug_mask_idxs, (batch_size, num_masked_spans * mask_length))

    offsets = tf.range(mask_length)[tf.newaxis, tf.newaxis, :]
    offsets = tf.tile(offsets, (batch_size, num_masked_spans, 1))
    offsets = tf.reshape(offsets, (batch_size, num_masked_spans * mask_length))

    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # scatter indices to mask
    spec_aug_mask = _scatter_values_on_batch_indices(
        tf.ones_like(spec_aug_mask_idxs), spec_aug_mask_idxs, tf.shape(
            spec_aug_mask)
    )

    return spec_aug_mask


def _get_feat_extract_output_lengths(input_lengths: tf.Tensor):
    """
    Computes the output length of the convolutional layers
    """

    def _conv_out_length(input_length, kernel_size, stride):
        # 1D convolutional layer output length formula taken
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        return (input_length - kernel_size) // stride + 1

    for kernel_size, stride in zip(config.conv_kernel, config.conv_stride):
        input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

    return input_lengths


def _mask_hidden_states(masked_spec_embed: tf.Tensor, hidden_states: tf.Tensor, mask_time_indices: Optional[tf.Tensor] = None):
    """
    Masks extracted features along time axis and/or along feature axis according to
    [SpecAugment](https://arxiv.org/abs/1904.08779).
    """
    batch_size, sequence_length, hidden_size = shape_list(hidden_states)

    # `config.apply_spec_augment` can set masking to False
    if not getattr(config, "apply_spec_augment", True):
        return hidden_states

    if mask_time_indices is not None:
        # apply SpecAugment along time axis with given mask_time_indices
        hidden_states = tf.where(
            tf.cast(mask_time_indices[:, :, tf.newaxis], tf.bool),
            masked_spec_embed[tf.newaxis, tf.newaxis, :],
            hidden_states,
        )

    elif config.mask_time_prob > 0:
        # generate indices & apply SpecAugment along time axis
        mask_time_indices = _compute_mask_indices(
            (batch_size, sequence_length),
            mask_prob=config.mask_time_prob,
            mask_length=config.mask_time_length,
            min_masks=2,
        )
        hidden_states = tf.where(
            tf.cast(mask_time_indices[:, :, tf.newaxis], tf.bool),
            masked_spec_embed[tf.newaxis, tf.newaxis, :],
            hidden_states,
        )

    # apply SpecAugment along feature axis
    if config.mask_feature_prob > 0:
        mask_feature_indices = _compute_mask_indices(
            (batch_size, hidden_size),
            mask_prob=config.mask_feature_prob,
            mask_length=config.mask_feature_length,
        )
        hidden_states = tf.where(
            mask_feature_indices[:, tf.newaxis, :], hidden_states, 0)

    return hidden_states


# Copied from transformers.models.bart.modeling_tf_bart._expand_mask
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


def wav2vec2_feature_encoder(input_values: tf.Tensor, training: bool = False, name: str = 'feature_extractor'):
    hidden_states = tf.expand_dims(input_values, -1)
    hidden_states = wav2vec2_groupnorm_conv_layer(
        layer_id=0)(hidden_states)

    for i in range(config.num_feat_extract_layers - 1):
        hidden_states = wav2vec2_no_layernorm_conv_layer(
            layer_id=i + 1)(hidden_states)
    return hidden_states


def wav2vec2_feature_projection(hidden_states: tf.Tensor, training: bool = False, name: str = 'feature_projection'):
    norm_hidden_states = layers.LayerNormalization(
        epsilon=config.layer_norm_eps)(hidden_states)
    hidden_states = layers.Dense(
        units=config.hidden_size,
        kernel_initializer=get_initializer(config.initializer_range),
        bias_initializer="zeros",
        name="projection",
    )(norm_hidden_states)
    hidden_states = layers.Dropout(
        rate=config.feat_proj_dropout)(hidden_states, training=training)
    return hidden_states, norm_hidden_states


def wav2vec2_encoder(
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        training: bool = False,
        return_dict: bool = True,
        name: str = 'encoder'):

    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None

    if attention_mask is not None:
        hidden_states = hidden_states * tf.expand_dims(attention_mask, -1)
        attention_mask = _expand_mask(attention_mask)
    else:
        attention_mask = None

    position_embeddings = wav2vec2_positional_conv_embedding(
        hidden_states)
    hidden_states = hidden_states + position_embeddings
    hidden_states = layers.LayerNormalization(
        epsilon=config.layer_norm_eps)(hidden_states)
    hidden_states = layers.Dropout(config.hidden_dropout)(hidden_states)

    for i in range(config.num_hidden_layers):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        dropout_probability = np.random.uniform(0, 1)
        if training and (dropout_probability < config.layerdrop):  # skip the layer
            continue

        layer_outputs = wav2vec2_encoder_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            training=training,
            name=f'layers.{i}'
        )
        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)

    # Add last layer
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

    return (
        hidden_states,
        all_hidden_states,
        all_self_attentions
    )


def wav2vec2_main_layer(inputs, mask_time_indices: tf.Tensor = None, name: str = None):
    attention_mask = inputs['attention_mask']
    extract_features = wav2vec2_feature_encoder(
        tf.cast(inputs["input_values"], tf.float32), training=inputs["training"]
    )
    # extract_features = tf.transpose(extract_features, perm=(0, 2, 1))

    if inputs["attention_mask"] is not None:
        # compute real output lengths according to convolution formula
        output_lengths = _get_feat_extract_output_lengths(
            tf.reduce_sum(inputs["attention_mask"], -1))

        attention_mask = tf.sequence_mask(
            output_lengths, maxlen=shape_list(extract_features)[
                1], dtype=extract_features.dtype
        )

    hidden_states, extract_features = wav2vec2_feature_projection(
        extract_features, training=inputs["training"])

    # mask_time_indices = kwargs.get("mask_time_indices", None)
    if inputs["training"]:
        hidden_states = _mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices)

    print('================== hidden',
          hidden_states.shape, extract_features.shape)
    print('before encoder', hidden_states.shape)
    encoder_outputs = wav2vec2_encoder(
        hidden_states,
        attention_mask=attention_mask,
        output_attentions=inputs["output_attentions"],
        output_hidden_states=inputs["output_hidden_states"],
        return_dict=inputs["return_dict"],
        training=inputs["training"],
    )
    hidden_states = encoder_outputs[0]

    if not inputs["return_dict"]:
        return (hidden_states, extract_features) + encoder_outputs[1:]

    return hidden_states


def wav2vec2_for_ctc(input_dim: int = 11200, vocab_size: int = 72):
    inputs = layers.Input(shape=(input_dim,), batch_size=1)
    print('input shape', inputs.shape)
    inputs_dict = {
        'input_values': inputs,
        'training': False,
        'attention_mask': None,
        'output_attentions': False,
        'output_hidden_states': False,
        'token_type_ids': None,
        'position_ids': None,
        'head_mask': None,
        'input_embeds': None,
        'return_dict': True,
        'labels': None
    }

    hidden_states = wav2vec2_main_layer(inputs_dict, name='wav2vec2')
    x = layers.Dropout(0.2)(hidden_states)
    outputs = layers.Dense(vocab_size, name='lm_head')(x)
    model = tf.keras.Model(inputs, outputs)
    return model
