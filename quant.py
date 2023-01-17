import tensorflow_model_optimization as tfmot

LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer


class TFWav2Vec2WeightNormConv1DQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
        return [
            (layer.kernel, LastValueQuantizer(num_bits=8,
             symmetric=True, narrow_range=False, per_axis=False)),
            (layer.weight_v, LastValueQuantizer(num_bits=8,
             symmetric=True, narrow_range=False, per_axis=False)),
            (layer.weight_g, LastValueQuantizer(num_bits=8,
             symmetric=True, narrow_range=False, per_axis=False)),
            (layer.bias, LastValueQuantizer(num_bits=8,
             symmetric=True, narrow_range=False, per_axis=False)),
        ]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in `get_weights_and_quantizers`
        # , in the same order
        layer.kernel = quantize_weights[0]
        layer.weight_w = quantize_weights[1]
        layer.weight_g = quantize_weights[2]

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        layer.activation = quantize_activations[0]

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}
