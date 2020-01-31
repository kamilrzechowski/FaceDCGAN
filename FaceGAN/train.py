from PIL import Image
import matplotlib.pyplot as plt
from .FaceGAN import FaceGAN

import numpy as np
import config as cfg
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

    plt.savefig(os.path.join(cfg.img_save_path, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.show()

def main():
    trainer = FaceGAN()
    trainer.train()

if __name__ == '__main__':
    main()
