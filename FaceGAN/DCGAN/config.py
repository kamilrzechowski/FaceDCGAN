#
# paths parameter
#

img_dir = 'img/*'
img_dir2 = 'C:/Kamil_VisulaStudio/FaceDCGAN/FaceGAN/img/*'
img_save_path = 'predictions/'
ckpt_generator = 'ckpt_generator/checkpoint'
ckpt_discriminator = 'ckpt_discriminator/checkpoint'
log_file = 'log.txt'


EPOCHS = 20
BATCH_SIZE = 64
noise_shape = (512,)
image_shape = (64,64,3)
num_examples_to_generate = 16