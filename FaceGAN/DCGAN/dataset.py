
import numpy as np
from . import config as cfg
from PIL import Image
from glob import glob
import os
import random

__all__ = ['LoadBatch']

class LoadBatch():

    def __init__(self):
        self.image_ids = glob(os.path.join(os.getcwd(), cfg.img_dir))

    def __getitem__(self, batch_no: int, batch=cfg.BATCH_SIZE):
        #shuffle the dataset if we start the epoch
        if batch_no == 0:
            random.shuffle(self.image_ids)

        crop = (30, 55, 150, 175)
        batch_ids = self.image_ids[batch_no*batch:(batch_no + 1)*batch]
        images = [np.array((Image.open(i).crop(crop)).resize((64,64))) for i in batch_ids]
        images = np.array(images)
        images = images/255
        images = images-0.5
    
        return np.array(images)

    

    