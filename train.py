import argparse

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from model import wav2vec2_for_ctc

quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quantize_scope = tfmot.quantization.keras.quantize_scope


def main():
    net = wav2vec2_for_ctc(input_dim=16000, vocab_size=45)
    # net.summary()

    quant_aware_model = tfmot.quantization.keras.quantize_apply(net)

    # converter = tf.lite.TFLiteConverter.from_keras_model(net)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.int8  # or tf.uint8
    # converter.inference_output_type = tf.int8  # or tf.uint8
    # tflite_model = converter.convert()
    # with open('model.tflite', 'wb') as f:
    #     f.write(tflite_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
