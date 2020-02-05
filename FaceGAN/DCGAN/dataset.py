
import numpy as np
from . import config as cfg
from PIL import Image
from glob import glob
import os
import random
import tensorflow as tf

__all__ = ['LoadBatch']

class LoadBatch():

    def __init__(self):
        self.image_ids = glob(os.path.join(os.getcwd(), cfg.img_dir2))

    def __getitem__(self, batch_no: int, batch=cfg.BATCH_SIZE):
        #shuffle the dataset if we start the epoch
        #if batch_no == 0:
        #    random.shuffle(self.image_ids)

        crop = (30, 55, 150, 175)
        batch_ids = self.image_ids[batch_no*batch:(batch_no + 1)*batch]
        images = [np.array((Image.open(i).crop(crop)).resize((64,64))) for i in batch_ids]
        images = np.array(images)
        images = images.reshape(images.shape[0], 64, 64, 3).astype('float32')
        images = (images - 127.5) / 127.5 # Normalize the images to [-1, 1]
        # Batch and shuffle the data
        images = tf.data.Dataset.from_tensor_slices(images) #.shuffle(cfg.BUFFER_SIZE).batch(cfg.BATCH_SIZE)

    
        return images #tf.convert_to_tensor(np.array(images), np.float32)

    def getitem(self, batch=cfg.BATCH_SIZE):
        BATCH_SIZE = 64
        IMG_HEIGHT = 64
        IMG_WIDTH = 64
        # The 1./255 is to convert from uint8 to float32 in range [0,1].
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        train_data_gen = image_generator.flow_from_directory(directory=cfg.img_dir2, batch_size=BATCH_SIZE, shuffle=False, target_size=(IMG_HEIGHT, IMG_WIDTH))

        return train_data_gen

    

    