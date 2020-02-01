# FaceDCGAN
 DCGAN for faces generation. Tensorflow 2.0  
   
I would like to thank the tensorflow team. Their sample code was very halpeful in creaing the project. https://www.tensorflow.org/tutorials/generative/dcgan  
 
## Instalation  
Prerequirements GPU tensorflow (computer with NVIDIA gpu):  
```
#create virtual enviromannt and install tensorflow gpu only
conda create -n tf-gpu tensorflow-gpu
conda activate tf-gpu
```
Prerequirements for CPU-only tensorflow:  
```
#create virtual enviromannt and install tensorflow cpu only
conda create -n tf tensorflow = 2.0
conda activate tf
```
  
Other library:  
```
#Install matplotlib
conda install -c conda-forge matplotlib
#Install tqdm
conda install -c conda-forge tqdm
#Install pillow
conda install -c anaconda pillow
```  
  
## Project structure  
```
 |-- FaceGAN/
     |-- ckpt_discriminator/
     |  |-- checkpoint1
     |  |   ......
     |  |-- checkpoint20
     |-- ckpt_generator/
     |  |-- checkpoint1
     |  |   ......
     |  |-- checkpoint20
     |-- DCGAN/
     |  |-- __init__.py
     |  |-- config.py
     |  |-- dataset.py
     |  |-- discriminator.py
     |  |-- FaceGAN.py
     |  |-- generator.py
     |-- img/
     |  |-- 000001.jpg
     |  |   ......
     |  |-- 200000.jpg
     |-- predictions/
     |  |-- image_at_epoch_1.png.jpg
     |  |   ......
     |  |-- image_at_epoch_20.png.jpg
     |-- train.py
```  

## Dataset  
The dataset used in the project is CelebA dataset of 200,000 images. The image size shuold be (64,64,3). If you want to use different image sizes, you will need to change network inputs to fit your requirements. The CelebA can be download from the webiste here: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. The dataset should be unzip to the /img folder inside the project.  
  
## Train  
To train the model place the CleabA dataset in the /img folder and run the train.py.  
  
## Predict  
TODO
