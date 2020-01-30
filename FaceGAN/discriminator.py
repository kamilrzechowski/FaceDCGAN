import tensorflow as tf
#from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2DTranspose, Conv2D, Input, Reshape, LeakyReLU, Activation
from tensorflow.keras import Model

__all__ = ['Discriminator']
EPSILON = 0.00005

class Discriminator():

    def make_deicriminator_model(in_shape):
        # 64*64*3 -> 32x32x64 
        inp = Input(in_shape)
        x = Conv2D(64, kernel_size=5, strides=2, padding='same')(inp)
        x = BatchNormalization(epsilon=EPSILON,trainable = True)(x)  
        x = LeakyReLU(0.2)(x)

        # 64*64*64 -> 64x64x128 
        x = Conv2D(128, kernel_size=5, strides=1, padding='same')(inp)
        x = BatchNormalization(epsilon=EPSILON,trainable = True)(x)  
        x = LeakyReLU(0.2)(x)
        
        # 64x64x128-> 32x32x256
        x = Conv2D(256, kernel_size=5, strides=2, padding='same')(inp)
        x = BatchNormalization(epsilon=EPSILON,trainable = True)(x)  
        x = LeakyReLU(0.2)(x)
        
        # 32x32x256 -> 16x16x512
        x = Conv2D(512, kernel_size=5, strides=2, padding='same')(inp)
        x = BatchNormalization(epsilon=EPSILON,trainable = True)(x)  
        x = LeakyReLU(0.2)(x)
        
        # 16x16x512 -> 8x8x1024
        x = Conv2D(1024, kernel_size=5, strides=1, padding='same')(inp)
        x = BatchNormalization(epsilon=EPSILON,trainable = True)(x)  
        x = LeakyReLU(0.2)(x)

        x = Reshape((-1,8*8*1024))(x)
        x = Dense(1)(x)
        out = Activation('sigmoid')(x)
    
        model = Model(inp, out)    
        
        return model
        

