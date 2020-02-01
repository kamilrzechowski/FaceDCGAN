#
# paths parameter
#

img_dir = 'img/*'
img_save_path = 'predictions/'
ckpt_generator = 'ckpt_generator/checkpoint'
ckpt_discriminator = 'ckpt_discriminator/checkpoint'


EPOCHS = 20
BATCH_SIZE = 128
noise_shape = (512,)
image_shape = (64,64,3)