import os
import time
import tensorflow as tf
import numpy as np
from glob import glob
import datetime
import random
from PIL import Image
import matplotlib.pyplot as plt

from tqdm import tqdm

from discriminator import Discriminator
from generator import Generator
import config as cfg

def get_images(batch_no: int, batch=64):
    image_ids = glob(cfg.img_dir)

    #shuffle the dataset if we start the epoch
    #if batch_no == 0:
    #    image_ids_shuffled = random.shuffle(image_ids)

    crop = (30, 55, 150, 175)
    batch_ids = image_ids[batch_no*batch:(batch_no + 1)*batch]
    images = [np.array((Image.open(i).crop(crop)).resize((64,64))) for i in batch_ids]
    images = np.array(images)
    images = images/255
    images = images-0.5
    
    return np.array(images)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join(cfg.img_save_path, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.show()

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, generator_module, discriminator_module, generator_optimizer, discriminator_optimizer, cross_entropy):
    noise = tf.random.normal([cfg.BATCH_SIZE, cfg.noise_dim])

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)   #should be 1 and what we recieved?
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)  #should be 0 and what we recieved?
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)    #we would like to foul the discriminator -> 1

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator_module(noise, training=True)

        real_output = discriminator_module(images, training=True)
        fake_output = discriminator_module(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator_module.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_module.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_module.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_module.trainable_variables))

def train(input_z_shape, input_x_shape, epochs: int,batch: int, checkpoint_path_gen = None, checkpoint_path_disc = None):
    generator_module = Generator.make_generator_model(input_z_shape)
    discriminator_module = Discriminator.make_deicriminator_model(input_x_shape)
    
    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    # num_examples_to_generate = batch
    seed = tf.random.normal([batch, input_z_shape[0]]) #[batch_size, noise_dim]

    image_ids = glob(cfg.img_dir)
    no_batch = len(image_ids)//batch

    if checkpoint_path_gen is not None:
        generator_module.load_weights(checkpoint_path_gen)
    if checkpoint_path_disc is not None:
        generator_module.load_weights(checkpoint_path_disc)

    
    for epoch in range(epochs):
        start = time.time()

        for i in tqdm(range(no_batch)):
            image_batch = get_images(i, batch=64)
            train_step(image_batch, generator_module, discriminator_module, generator_optimizer, discriminator_optimizer, cross_entropy)

        # Save the model every 15 epochs
        if (epoch + 1) % 2 == 0:
            #checkpoint.save(file_prefix = checkpoint_prefix)
            generator_module.save_weights(cfg.ckpt_generator + str(epoch))
            discriminator_module.save_weights(cfg.ckpt_discriminator + str(epoch))

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator_module, epochs, seed)


def main():
    train((cfg.noise_dim,), (64,64,3), cfg.EPOCHS, cfg.BATCH_SIZE)

if __name__ == '__main__':
    main()