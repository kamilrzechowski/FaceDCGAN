import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2DTranspose, Conv2D, Input, Reshape, LeakyReLU, Activation
from tensorflow.keras import Model

__all__ = ['Generator']
EPSILON = 0.00005

class Generator():

    def make_generator_model(in_shape):
        # 8x8x1024
        inp = Input(in_shape)
        x = Dense(8*8*1024,use_bias=False)(inp)
        x = Reshape((8, 8, 1024))(x)
        x = LeakyReLU(0.2)(x)
        #assert x.output_shape  == (None, 8, 8, 1024) # Note: None is the batch size

        # 8x8x1024 -> 16x16x512
        x = Conv2DTranspose(512, kernel_size=5, strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization(epsilon=EPSILON)(x)        
        x = LeakyReLU(0.2)(x)
        
        # 16x16x512 -> 32x32x256
        x = Conv2DTranspose(256, kernel_size=5, strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization(epsilon=EPSILON)(x)        
        x = LeakyReLU(0.2)(x)
        
        # 32x32x256 -> 64x64x128
        x = Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization(epsilon=EPSILON)(x)        
        x = LeakyReLU(0.2)(x)
        
        # 64x64x128 -> 128x128x64 64x64x3
        x = Conv2DTranspose(3, kernel_size=5, strides=1, padding='same', use_bias=False)(x)
        out = Activation('tanh')(x)
        #assert out.output_shape == (None, 64, 64, 3)
        
        model = Model(inp, out)    
        
        return model



    '''
    https://www.tensorflow.org/tutorials/generative/dcgan#next_steps
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=in_shape)) #(100,)  #z.get_shape().as_list()[-1]
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())'''