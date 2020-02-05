from __future__ import absolute_import, division

import time
import tensorflow as tf
import numpy as np
import datetime
import random
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import os
import imageio
import PIL

from IPython import display

from .discriminator import Discriminator
from .generator import Generator
from .dataset import LoadBatch
from . import config as cfg

__all__ = ['FaceGAN']

class FaceGAN():

    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.generator_module = self.generator.make_generator_model(cfg.noise_shape)
        self.discriminator_module = self.discriminator.make_deicriminator_model(cfg.image_shape)

        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.batchLoader = LoadBatch()

    def _discriminator_loss(self,real_output, fake_output):
            real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)   #should be 1 and what we recieved?
            fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)  #should be 0 and what we recieved?
            total_loss = real_loss + fake_loss
            return total_loss

    def _generator_loss(self,fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)    #we would like to foul the discriminator -> 1

    def _get_no_batches(self):
        return len(glob(os.path.join(os.getcwd(), cfg.img_dir2)))//cfg.BATCH_SIZE


    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self,images):
        noise = tf.random.normal([cfg.BATCH_SIZE, cfg.noise_shape[0]])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator_module(noise, training=True)

            real_output = self.discriminator_module(images, training=True)
            fake_output = self.discriminator_module(generated_images, training=True)

            gen_loss = self._generator_loss(fake_output)
            disc_loss = self._discriminator_loss(real_output, fake_output)
            #tf.print("Gen loss=", gen_loss, ", Disc loss=", disc_loss)
            tf.print("Gen loss=", gen_loss, ", Disc loss=", disc_loss, 
                     output_stream = os.path.join('file://',cfg.log_file))

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator_module.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator_module.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator_module.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator_module.trainable_variables))

    def generate_and_save_images(self, model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        #fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(os.path.join(os.path.join(os.getcwd(), cfg.img_save_path), 'image_at_epoch_{:04d}.png'.format(epoch)))
        #plt.show()

    def train(self, checkpoint_path_gen = None, checkpoint_path_disc = None):

        # We will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        # num_examples_to_generate = batch
        seed = tf.random.normal([cfg.num_examples_to_generate, cfg.noise_shape[0]]) #[batch_size, noise_dim]

        no_batch = self._get_no_batches()

        if checkpoint_path_gen is not None:
            self.generator_module.load_weights(checkpoint_path_gen)
        if checkpoint_path_disc is not None:
            self.generator_module.load_weights(checkpoint_path_disc)


        for epoch in range(cfg.EPOCHS):
            start = time.time()

            for i in tqdm(range(no_batch)):
                for minibatch in self.batchLoader[i]:
                    self.train_step(minibatch)

            # Save the model every 15 epochs
            if (epoch + 1) % 2 == 0:
                #checkpoint.save(file_prefix = checkpoint_prefix)
                self.generator_module.save_weights(os.path.join(os.getcwd(), cfg.ckpt_generator) + str(epoch))
                self.discriminator_module.save_weights(os.path.join(os.getcwd(), cfg.ckpt_discriminator) + str(epoch))

            # Produce images for the GIF as we go
            display.clear_output(wait=True)
            self.generate_and_save_images(self.generator_module, epoch + 1, seed)

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            tf.print("Epoch=", epoch, ", time=", time.time()-start,output_stream = os.path.join('file://',cfg.log_file))

        #Generate after the final epoch
        #display.clear_output(wait=True)
        #generate_and_save_images(self.generator_module, epochs, seed)


