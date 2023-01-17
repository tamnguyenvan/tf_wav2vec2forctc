from typing import Optional, Tuple
import tensorflow as tf

from utils import shape_list, stable_softmax


def wav2vec2_weight_norm_conv1d(filters: int, kernel_size: int, groups: int, explicit_padding: int):
    def _wav2vec2_weight_norm_conv1d(inputs: tf.Tensor):
        # Build
        input_shape = inputs.shape
        input_shape = input_shape.as_list()
        filter_axis = 2
        kernel_norm_axes = tf.constant([0, 1])

        # Conv1D output shapes are checked at build time since TF 2.7, so we need to account for padding
        input_shape[-2] += explicit_padding * 2

        # super
        base_conv1d = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            groups=groups,
            padding="valid",
            use_bias=True,
            bias_initializer="he_normal",
        )
        base_conv1d.build(input_shape)

        base_conv1d.kernel = tf.Variable(tf.transpose(
            base_conv1d.kernel), name="weight_v", trainable=True)
        base_conv1d.weight_v = base_conv1d.kernel
        base_conv1d.weight_g = base_conv1d.add_weight(
            name="weight_g",
            shape=(int(base_conv1d.weight_v.shape[filter_axis]), 1, 1),
            initializer="ones",
            dtype=base_conv1d.weight_v.dtype,
            trainable=True,
        )
        base_conv1d.bias = base_conv1d.add_weight(
            name="bias",
            shape=(filters,),
            initializer="zeros",
            trainable=True
        )

        """Set the norm of the weight vector."""
        kernel_norm = tf.sqrt(tf.reduce_sum(
            tf.square(base_conv1d.weight_v), axis=kernel_norm_axes))
        base_conv1d.weight_g.assign(kernel_norm[:, tf.newaxis, tf.newaxis])

        # Normalize kernel
        kernel = tf.nn.l2_normalize(
            base_conv1d.weight_v, axis=kernel_norm_axes) * tf.transpose(base_conv1d.weight_g)
        base_conv1d.kernel = tf.transpose(kernel)

        import pdb
        pdb.set_trace()
        padded_inputs = tf.pad(
            inputs, ((0, 0), (explicit_padding, explicit_padding), (0, 0)))
        outputs = base_conv1d(padded_inputs)
        return outputs
    return _wav2vec2_weight_norm_conv1d


def wav2vec2_group_norm(
    groups: int = 32,
    axis: int = -1,
    epsilon: float = 1e-3,
    center: bool = True,
    scale: bool = True,
    beta_initializer: tf.keras.initializers.Initializer = "zeros",
    gamma_initializer: tf.keras.initializers.Initializer = "ones",
    beta_regularizer: tf.keras.regularizers.Regularizer = None,
    gamma_regularizer: tf.keras.regularizers.Regularizer = None,
    beta_constraint: tf.keras.constraints.Constraint = None,
    gamma_constraint: tf.keras.constraints.Constraint = None,
    **kwargs,
):

    def _reshape_into_groups(layer, inputs, input_shape, tensor_input_shape):

        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        is_instance_norm = (input_shape[layer.axis] // layer.groups) == 1
        if not is_instance_norm:
            group_shape[layer.axis] = input_shape[layer.axis] // layer.groups
            group_shape.insert(layer.axis, layer.groups)
            group_shape = tf.stack(group_shape)
            reshaped_inputs = tf.reshape(inputs, group_shape)
            return reshaped_inputs, group_shape
        else:
            return inputs, group_shape

    def _apply_normalization(layer, reshaped_inputs, input_shape):

        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        is_instance_norm = (input_shape[layer.axis] // layer.groups) == 1
        if not is_instance_norm:
            axis = -2 if layer.axis == -1 else layer.axis - 1
        else:
            axis = -1 if layer.axis == -1 else layer.axis - 1
        group_reduction_axes.pop(axis)

        mean, variance = tf.nn.moments(
            reshaped_inputs, group_reduction_axes, keepdims=True)

        gamma, beta = _get_reshaped_weights(layer, input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=epsilon,
        )
        return normalized_inputs

    def _get_reshaped_weights(layer, input_shape):
        broadcast_shape = _create_broadcast_shape(layer, input_shape)
        gamma = None
        beta = None
        if scale:
            gamma = tf.reshape(layer.gamma, broadcast_shape)

        if center:
            beta = tf.reshape(layer.beta, broadcast_shape)
        return gamma, beta

    def _check_if_input_shape_is_none(layer, input_shape):
        dim = input_shape[layer.axis]
        if dim is None:
            raise ValueError(
                "Axis "
                + str(axis)
                + " of input tensor should have a defined dimension but the layer received an input with shape "
                + str(input_shape)
                + "."
            )

    def _set_number_of_groups_for_instance_norm(layer, input_shape):
        dim = input_shape[layer.axis]

        if layer.groups == -1:
            layer.groups = dim

    def _check_size_of_dimensions(layer, input_shape):

        dim = input_shape[layer.axis]
        if dim < layer.groups:
            raise ValueError(
                "Number of groups ("
                + str(layer.groups)
                + ") cannot be more than the number of channels ("
                + str(dim)
                + ")."
            )

        if dim % layer.groups != 0:
            raise ValueError(
                "Number of groups ("
                + str(layer.groups)
                + ") must be a multiple of the number of channels ("
                + str(dim)
                + ")."
            )

    def _check_axis(layer):

        if layer.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to use tf.layer.batch_normalization instead"
            )

    def _create_input_spec(layer, input_shape):

        dim = input_shape[layer.axis]
        layer.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes={layer.axis: dim})

    def _add_gamma_weight(layer, input_shape):

        dim = input_shape[layer.axis]
        shape = (dim,)

        if scale:
            layer.gamma = layer.add_weight(
                shape=shape,
                name="gamma",
                initializer=layer.gamma_initializer,
                regularizer=layer.gamma_regularizer,
                constraint=layer.gamma_constraint,
            )
        else:
            layer.gamma = None

    def _add_beta_weight(layer, input_shape):

        dim = input_shape[layer.axis]
        shape = (dim,)

        if center:
            layer.beta = layer.add_weight(
                shape=shape,
                name="beta",
                initializer=layer.beta_initializer,
                regularizer=layer.beta_regularizer,
                constraint=layer.beta_constraint,
            )
        else:
            layer.beta = None

    def _create_broadcast_shape(layer, input_shape):
        broadcast_shape = [1] * len(input_shape)
        is_instance_norm = (input_shape[layer.axis] // layer.groups) == 1
        if not is_instance_norm:
            broadcast_shape[layer.axis] = input_shape[layer.axis] // layer.groups
            broadcast_shape.insert(layer.axis, layer.groups)
        else:
            broadcast_shape[layer.axis] = layer.groups
        return broadcast_shape

    def _wav2vec2_group_norm(inputs: tf.Tensor):
        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)

        # Build
        base_layer = tf.keras.layers.Layer()
        base_layer.supports_masking = True
        base_layer.groups = groups
        base_layer.axis = axis
        base_layer.epsilon = epsilon
        base_layer.center = center
        base_layer.scale = scale
        base_layer.beta_initializer = tf.keras.initializers.get(
            beta_initializer)
        base_layer.gamma_initializer = tf.keras.initializers.get(
            gamma_initializer)
        base_layer.beta_regularizer = tf.keras.regularizers.get(
            beta_regularizer)
        base_layer.gamma_regularizer = tf.keras.regularizers.get(
            gamma_regularizer)
        base_layer.beta_constraint = tf.keras.constraints.get(beta_constraint)
        base_layer.gamma_constraint = tf.keras.constraints.get(
            gamma_constraint)
        _check_axis(base_layer)

        _check_if_input_shape_is_none(base_layer, input_shape)
        _set_number_of_groups_for_instance_norm(base_layer, input_shape)
        _check_size_of_dimensions(base_layer, input_shape)
        _create_input_spec(base_layer, input_shape)

        _add_gamma_weight(base_layer, input_shape)
        _add_beta_weight(base_layer, input_shape)
        base_layer.build(input_shape)

        reshaped_inputs, group_shape = _reshape_into_groups(
            base_layer, inputs, input_shape, tensor_input_shape)

        normalized_inputs = _apply_normalization(
            base_layer, reshaped_inputs, input_shape)

        is_instance_norm = (
            input_shape[base_layer.axis] // base_layer.groups) == 1
        if not is_instance_norm:
            outputs = tf.reshape(normalized_inputs, tensor_input_shape)
        else:
            outputs = normalized_inputs

        return outputs

    return _wav2vec2_group_norm


def wav2vec2_attention(
    embed_dim: int,
    num_heads: int,
    dropout: float = 0.0,
    is_decoder: bool = False,
    bias: bool = True,
    **kwargs,
):
    def _shape(layer, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, layer.num_heads, layer.head_dim)), (0, 2, 1, 3))

    def _wav2vec2_attention(
        hidden_states: tf.Tensor,
        key_value_states: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[tf.Tensor]]] = None,
        attention_mask: Optional[tf.Tensor] = None,
        layer_head_mask: Optional[tf.Tensor] = None,
        training: Optional[bool] = False,
    ):
        # # Init
        base_layer = tf.keras.layers.Layer()
        base_layer.embed_dim = embed_dim

        base_layer.num_heads = num_heads
        base_layer.dropout = tf.keras.layers.Dropout(dropout)
        base_layer.head_dim = embed_dim // num_heads
        if (base_layer.head_dim * num_heads) != base_layer.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {base_layer.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        base_layer.scaling = base_layer.head_dim**-0.5
        base_layer.is_decoder = is_decoder

        base_layer.k_proj = tf.keras.layers.Dense(
            embed_dim, use_bias=bias)
        base_layer.q_proj = tf.keras.layers.Dense(
            embed_dim, use_bias=bias)
        base_layer.v_proj = tf.keras.layers.Dense(
            embed_dim, use_bias=bias)
        base_layer.out_proj = tf.keras.layers.Dense(
            embed_dim, use_bias=bias)

        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim_ = shape_list(hidden_states)

        # get query proj
        query_states = base_layer.q_proj(hidden_states) * base_layer.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = _shape(
                base_layer, base_layer.k_proj(key_value_states), -1, bsz)
            value_states = _shape(
                base_layer, base_layer.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = _shape(
                base_layer, base_layer.k_proj(hidden_states), -1, bsz)
            value_states = _shape(
                base_layer, base_layer.v_proj(hidden_states), -1, bsz)
            key_states = tf.concat([past_key_value[0], key_states], axis=2)
            value_states = tf.concat([past_key_value[1], value_states], axis=2)
        else:
            # self_attention
            key_states = _shape(
                base_layer, base_layer.k_proj(hidden_states), -1, bsz)
            value_states = _shape(
                base_layer, base_layer.v_proj(hidden_states), -1, bsz)

        if base_layer.is_decoder:
            # if cross_attention save Tuple(tf.Tensor, tf.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(tf.Tensor, tf.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * base_layer.num_heads, -1, base_layer.head_dim)
        query_states = tf.reshape(_shape(
            base_layer, query_states, tgt_len, bsz), proj_shape)
        key_states = tf.reshape(key_states, proj_shape)
        value_states = tf.reshape(value_states, proj_shape)

        src_len = shape_list(key_states)[1]
        attn_weights = tf.matmul(query_states, key_states, transpose_b=True)

        tf.debugging.assert_equal(
            shape_list(attn_weights),
            [bsz * base_layer.num_heads, tgt_len, src_len],
            message=(
                f"Attention weights should be of size {(bsz * base_layer.num_heads, tgt_len, src_len)}, but is"
                f" {shape_list(attn_weights)}"
            ),
        )

        if attention_mask is not None:
            tf.debugging.assert_equal(
                shape_list(attention_mask),
                [bsz, 1, tgt_len, src_len],
                message=(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {shape_list(attention_mask)}"
                ),
            )

            attention_mask = tf.cast(attention_mask, dtype=attn_weights.dtype)
            attn_weights = tf.reshape(
                attn_weights, (bsz, base_layer.num_heads, tgt_len, src_len)) + attention_mask
            attn_weights = tf.reshape(
                attn_weights, (bsz * base_layer.num_heads, tgt_len, src_len))

        attn_weights = stable_softmax(attn_weights, axis=-1)

        if layer_head_mask is not None:
            tf.debugging.assert_equal(
                shape_list(layer_head_mask),
                [base_layer.num_heads],
                message=(
                    f"Head mask for a single layer should be of size {(base_layer.num_heads)}, but is"
                    f" {shape_list(layer_head_mask)}"
                ),
            )

            attn_weights = tf.reshape(layer_head_mask, (1, -1, 1, 1)) * tf.reshape(
                attn_weights, (bsz, base_layer.num_heads, tgt_len, src_len)
            )
            attn_weights = tf.reshape(
                attn_weights, (bsz * base_layer.num_heads, tgt_len, src_len))

        attn_probs = base_layer.dropout(attn_weights, training=training)
        attn_output = tf.matmul(attn_probs, value_states)

        tf.debugging.assert_equal(
            shape_list(attn_output),
            [bsz * base_layer.num_heads, tgt_len, base_layer.head_dim],
            message=(
                f"`attn_output` should be of size {(bsz, base_layer.num_heads, tgt_len, base_layer.head_dim)}, but is"
                f" {shape_list(attn_output)}"
            ),
        )

        attn_output = tf.transpose(
            tf.reshape(attn_output, (bsz, base_layer.num_heads,
                       tgt_len, base_layer.head_dim)), (0, 2, 1, 3)
        )
        attn_output = tf.reshape(attn_output, (bsz, tgt_len, embed_dim_))

        attn_output = base_layer.out_proj(attn_output)
        attn_weights: tf.Tensor = tf.reshape(
            attn_weights, (bsz, base_layer.num_heads, tgt_len, src_len))

        return attn_output, attn_weights, past_key_value
    return _wav2vec2_attention
