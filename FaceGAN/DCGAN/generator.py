import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2DTranspose, Conv2D, Input, Reshape, LeakyReLU, Activation
from tensorflow.keras import Model

__all__ = ['Generator']
EPSILON = 0.00005

class Generator():

    def make_generator_model(self, in_shape):
        model = tf.keras.Sequential()
        model.add(layers.Dense(8*8*512, use_bias=False, input_shape=(512,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((8, 8, 512)))
        assert model.output_shape == (None, 8, 8, 512) # Note: None is the batch size

        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 256)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 32, 32, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 64, 64, 3)

        return model