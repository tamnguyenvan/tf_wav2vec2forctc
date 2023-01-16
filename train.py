import argparse

import tensorflow as tf

from model import wav2vec2_for_ctc
from data_loader import create_data_loader


def main():
    create_data_loader()
    # net = wav2vec2_for_ctc(input_dim=11200, vocab_size=45)
    # net.summary()

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
