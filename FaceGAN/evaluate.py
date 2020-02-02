from __future__ import absolute_import

import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from DCGAN.generator import Generator
from DCGAN import config as cfg

import numpy as np
from PIL import Image
from glob import glob
import os
import random

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join(os.path.join(os.getcwd(), cfg.img_save_path), 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.show()

def main():
    generator_model = Generator.make_generator_model(cfg.noise_shape)
    generator_model.load_weights("C:/Kamil_VisulaStudio/FaceDCGAN/FaceGAN/ckpt_generator/checkpoint5")
    noise = tf.random.normal([cfg.BATCH_SIZE, cfg.noise_shape[0]])
    fake_image = generator_model.predict(noise)

    print("max value: " + str(np.amax(fake_image[:,:])))
    print("min value: " + str(np.amin(fake_image[:,:])))
    int_fake_image = (fake_image[0] + 0.5)*255
    print("max value: " + str(np.amax(int_fake_image[:,:])))
    print("min value: " + str(np.amin(int_fake_image[:,:])))
    plt.imshow(int_fake_image)
    plt.show()



if __name__ == '__main__':
    main()

