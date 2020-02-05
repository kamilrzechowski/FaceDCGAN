from __future__ import absolute_import

from PIL import Image
import matplotlib.pyplot as plt
from DCGAN.FaceGAN import FaceGAN
from DCGAN import config as cfg

import numpy as np
from PIL import Image
from glob import glob
import os
import random

def main():
    trainer = FaceGAN()
    trainer.train()

if __name__ == '__main__':
    main()
