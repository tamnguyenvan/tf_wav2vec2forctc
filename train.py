import argparse

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from model import wav2vec2_for_ctc
# from ops import TFWav2Vec2WeightNormConv1D
from quant import (
    # TFWav2Vec2WeightNormConv1DQuantizeConfig,
    Conv1DQuantizeConfig
)

quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quantize_scope = tfmot.quantization.keras.quantize_scope

import tf2onnx


def apply_quantization(layer):
    if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
        return quantize_annotate_layer(layer)
    return layer


def main():
    net = wav2vec2_for_ctc(input_dim=11200, vocab_size=45)
    # net.save('./saved_model')
    # net.summary()
    # net = tf.keras.models.clone_model(net, clone_function=apply_quantization)

    # quant_scope = {
    #     # 'TFWav2Vec2WeightNormConv1D': TFWav2Vec2WeightNormConv1D,
    #     'Conv1DQuantizeConfig': Conv1DQuantizeConfig
    # }
    # with quantize_scope(quant_scope):
    #     quant_aware_model = tfmot.quantization.keras.quantize_apply(net)

    # quant_aware_model.summary()

    # Training here
    # converter = tf.lite.TFLiteConverter.from_keras_model(net)
    # # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # tflite_model = converter.convert()
    # with open('wav2vec2.tflite', 'wb') as f:
    #     f.write(tflite_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
