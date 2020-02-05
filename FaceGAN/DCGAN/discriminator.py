import tensorflow as tf
#from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2DTranspose, Conv2D, Input, Reshape, LeakyReLU, Activation
from tensorflow.keras import Model

__all__ = ['Discriminator']
EPSILON = 0.00005

class Discriminator():

    def make_deicriminator_model(self, in_shape):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                         input_shape=[64, 64, 3]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))  #1-> output size

        return model
        

