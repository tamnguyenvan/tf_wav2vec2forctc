from typing import Union, List, Optional
import warnings
import logging
from typing import Mapping

import inspect
import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)


def booleans_processing(config, **kwargs):
    """
    Process the input booleans of each model.
    Args:
        config ([`PretrainedConfig`]):
            The config of the running model.
        **kwargs:
            The boolean parameters
    Returns:
        A dictionary with the proper values for each boolean
    """
    final_booleans = {}

    # Pure conv models (such as ConvNext) do not have `output_attentions`. If the signature has
    # `output_attentions`, it will be present here in `kwargs`, even if unset (in that case, as `None`)
    if "output_attentions" in kwargs:
        final_booleans["output_attentions"] = (
            kwargs["output_attentions"] if kwargs["output_attentions"] is not None else config.output_attentions
        )
    final_booleans["output_hidden_states"] = (
        kwargs["output_hidden_states"] if kwargs["output_hidden_states"] is not None else config.output_hidden_states
    )
    final_booleans["return_dict"] = kwargs["return_dict"] if kwargs["return_dict"] is not None else config.return_dict

    if "use_cache" in kwargs:
        final_booleans["use_cache"] = (
            kwargs["use_cache"] if kwargs["use_cache"] is not None else getattr(
                config, "use_cache", None)
        )
    return final_booleans


def input_values_processing(func, config, input_values, **kwargs):
    """
    Process the input of each TensorFlow model including the booleans. In case of a list of symbolic inputs, each input
    has to be named accordingly to the parameters name, i.e. `input_values = tf.keras.Input(shape=(128,),
    dtype='float32', name="input_values")` otherwise the order of the tensors will not be guaranteed during the
    training.
    Args:
        func (`callable`):
            The callable function of the TensorFlow model.
        config ([`PretrainedConfig`]):
            The config of the running model.
        **kwargs:
            The inputs of the model.
    Returns:
        Two lists, one for the missing layers, and another one for the unexpected layers.
    """
    signature = dict(inspect.signature(func).parameters)
    signature.pop("kwargs", None)
    signature.pop("self", None)
    parameter_names = list(signature.keys())
    output = {}
    allowed_types = (tf.Tensor, bool, int,
                     tuple, list, dict, np.ndarray)

    for k, v in kwargs.items():
        if isinstance(v, allowed_types) or v is None:
            output[k] = v
        else:
            raise ValueError(
                f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")

    if isinstance(input_values, (tuple, list)):
        for i, input in enumerate(input_values):
            # EagerTensors don't allow to use the .name property so we check for a real Tensor
            if type(input) == tf.Tensor:
                # Tensor names have always the pattern `name:id` then we check only the
                # `name` part
                tensor_name = input.name.split(":")[0]

                if tensor_name in parameter_names:
                    output[tensor_name] = input
                else:
                    output[parameter_names[i]] = input
            elif isinstance(input, allowed_types) or input is None:
                output[parameter_names[i]] = input
            else:
                raise ValueError(
                    f"Data of type {type(input)} is not allowed only {allowed_types} is accepted for"
                    f" {parameter_names[i]}."
                )
    elif isinstance(input_values, Mapping):
        if "inputs" in input_values:
            warnings.warn(
                "The `inputs` argument is deprecated and will be removed in a future version, use `input_values`"
                " instead.",
                FutureWarning,
            )

            output["input_values"] = input_values.pop("inputs")

        if "decoder_cached_states" in input_values:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use"
                " `past_key_values` instead.",
                FutureWarning,
            )
            output["past_key_values"] = input_values.pop(
                "decoder_cached_states")

        for k, v in dict(input_values).items():
            if isinstance(v, allowed_types) or v is None:
                output[k] = v
            elif k not in parameter_names and "args" not in parameter_names:
                logger.warning(
                    f"The parameter {k} does not belongs to the parameter list {parameter_names} and will be ignored."
                )
                continue
            else:
                raise ValueError(
                    f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")
    else:
        if isinstance(input_values, tf.Tensor) or input_values is None:
            output[parameter_names[0]] = input_values
        else:
            raise ValueError(
                f"Data of type {type(input_values)} is not allowed only {allowed_types} is accepted for"
                f" {parameter_names[0]}."
            )

    for name in parameter_names:
        if name not in list(output.keys()) and name != "args":
            output[name] = kwargs.pop(name, signature[name].default)

    # When creating a SavedModel TF calls the method with LayerCall.__call__(args, **kwargs)
    # So to respect the proper output we have to add this exception
    if "args" in output:
        if output["args"] is not None and type(output["args"]) == tf.Tensor:
            tensor_name = output["args"].name.split(":")[0]
            output[tensor_name] = output["args"]
        else:
            # `args` in this case is always the first parameter, then `input_values`
            output["input_values"] = output["args"]

        del output["args"]

    if "kwargs" in output:
        del output["kwargs"]

    boolean_dict = {
        k: v
        for k, v in output.items()
        if k in ["return_dict", "output_attentions", "output_hidden_states", "use_cache"]
    }

    output.update(booleans_processing(config=config, **boolean_dict))

    return output


def shape_list(tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.
    Args:
        tensor (`tf.Tensor` or `np.ndarray`): The tensor we want the shape of.
    Returns:
        `List[int]`: The shape of the tensor as a list.
    """
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def stable_softmax(logits: tf.Tensor, axis: Optional[int] = None, name: Optional[str] = None) -> tf.Tensor:
    """
    Stable wrapper that returns the same output as `tf.nn.softmax`, but that works reliably with XLA on CPU. It is
    meant as a workaround for the [following issue](https://github.com/tensorflow/tensorflow/issues/55682), and will be
    removed after it gets fixed. The arguments and outputs are the same as `tf.nn.softmax`, and relies on the fact that
    `softmax(x) = softmax(x + c)` (see https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html).
    Args:
        logits (`tf.Tensor`):
            Must be one of the following types: half, float32, float64.
        axis (`int`, *optional*):
            The dimension softmax would be performed on. The default is -1 which indicates the last dimension.
        name (`str`, *optional*):
            A name for the operation.
    Returns:
        `tf.Tensor`:
            A Tensor. Has the same type and shape as logits.
    """
    # TODO: When the issue linked above gets sorted, add a check on TF version here and use the original function if
    # it has the fix. After we drop the support for unfixed versions, remove this function.
    return tf.nn.softmax(logits=logits + 1e-9, axis=axis, name=name)


def get_initializer(initializer_range: float = 0.02) -> tf.initializers.TruncatedNormal:
    """
    Creates a `tf.initializers.TruncatedNormal` with the given range.
    Args:
        initializer_range (*float*, defaults to 0.02): Standard deviation of the initializer range.
    Returns:
        `tf.initializers.TruncatedNormal`: The truncated normal initializer.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
