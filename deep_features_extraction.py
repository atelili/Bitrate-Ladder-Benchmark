"""
Usage:
    python features_extration -v (video directory) 
    -f (csv file) -o (overlapping between patches , default = 0.2) 
    -np (num patches, default=25) -nf (num frames, default=30)
    -m (backbone model, default resnet50) 
Author : 
    Ahmed Telili
"""




import numpy as np
import cv2
import os 
from tensorflow.keras import Input
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.models import Model
from sklearn.utils import shuffle
from tensorflow import keras
import pandas as pd
import csv
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications 
import PIL
import h5py
from PIL import Image
from keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.layers import Dense ,Dropout ,Input,concatenate,Conv2D,Reshape,GlobalMaxPooling2D,Flatten,GlobalAveragePooling2D
import argparse
import random
from tqdm import tqdm




tf.keras.backend.clear_session()

def start_points(size, split_size, overlap=0):
	points = [0]
	stride = int(split_size * (1-overlap))
	counter = 1
	while True:
		pt = stride * counter
		if pt + split_size >= size:
			points.append(size - split_size)
			break
		else:
			points.append(pt)
			counter += 1
	return points

def random_crop(img, shape):
	return tf.image.random_crop(img, shape)

def crop_image(img, overlapping,num_patch):

	img_h, img_w, _ = img.shape
	split_width = 224
	split_height = 224
	X_points = start_points(img_w, split_width, overlapping)
	Y_points = start_points(img_h, split_height,overlapping )

	count = 0
	imgs = []


	for i in Y_points:
		for j in X_points:
			split = img[i:i+split_height, j:j+split_width]
			imgs.append(split)
			count += 1



	if len(X_points)*len(Y_points) < num_patch:
		dif = num_patch - len(X_points)*len(Y_points)
		for i in range(dif) :
			imgs.append(random_crop(img,(224,224,3)).numpy())

	elif len(X_points)*len(Y_points) > num_patch:
		imgs = imgs[0:num_patch]

	




	return(imgs)

def TemporalCrop(rgbs,  nb):
  final = []
  step = int(64/nb)

  i = 0
  j = 0
  while i < nb :
    img = rgbs[j]
    final.append(img)
    j = j +step
    i = i +1    
  return(final)








class DataGenerator(keras.utils.Sequence):
    def __init__(self, patches,list_IDs,overlapping, nb, backbone, 
    				shuffle=False, batch_size=1 ):
        'Initialization'
        self.batch_size = batch_size
        self.nb = nb
        self.backbone = backbone
        self.patches = patches
        self.shuffle = shuffle
        self.list_IDs = list_IDs
        self.overlapping = overlapping
        self.on_epoch_end()
   
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs)/ self.batch_size))

    def __getitem__(self, index):
        
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        batch = [self.list_IDs[k] for k in indexes]
        
        # Generate data
        X, y, names = self.__data_generation(batch, self.nb, self.backbone)

        return  X, y, names

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, batch, nb, backbone):

        # Initialization
        X = np.empty((self.nb, self.patches,224,224, 3))
        
        y = np.empty((self.batch_size,3), dtype=np.float32)

        
        # Generate data
        for i, ID in enumerate(batch):
            print(batch)
            imgs_init, names = read_yuv(ID[0])

            imgs = TemporalCrop(imgs_init, self.nb)

            for k in range(len(imgs)):
            	im = crop_image(imgs[k],overlapping= self.overlapping, num_patch= self.patches)
            	for j in range(self.patches):
            		im = np.array(im)
            		if self.backbone == 'resnet50':
            			X[k,j,:,:,:]=tf.keras.applications.resnet50.preprocess_input(im[j,:,:,:])
            		elif self.backbone == 'vgg16':
            			X[k,j,:,:,:]=tf.keras.applications.vgg16.preprocess_input(im[j,:,:,:])
            		elif self.backbone == 'densenet169':
            			X[k,j,:,:,:]=tf.keras.applications.densenet.preprocess_input(im[j,:,:,:])
            		elif self.backbone == 'inception_v3':
            			X[k,j,:,:,:]=tf.keras.applications.inception_v3.preprocess_input(im[j,:,:,:])
            		else:
            			X[k,j,:,:,:]=tf.keras.applications.resnet50.preprocess_input(im[j,:,:,:])

            names = ID[0].split('/')[-1]	

            y[i] = ID[1:]	
		
                   
              
        return X, y, names



def prepare_datalist(path_to_csv, videos_dir):
	data1 = pd.read_csv(path_to_csv)
	li = data1.values.tolist()
	li.sort()
	for i in range(len(li)):
		li[i][0] = videos_dir + '/' + str(li[i][0]) +'_3840x2160_8bit_420_60fps_frames1-64.yuv'
		#li[i][1] =  li[i][1].replace(",", ".")
		#li[i][2] =  li[i][2].replace(",", ".")
		#li[i][3] =  li[i][3].replace(",", ".")
	
	return(li)

def read_yuv(video_path):
    v = video_path.split('/')[-1]

    gray_frames = []
    rgb_frames = []
    file_size = os.path.getsize(video_path)
    names = v.split('_')[0]
    resolution = v.split('_')[1]
    width = int(resolution.split('x')[0])
    height = int(resolution.split('x')[1])
    n_frames = file_size // (width*height*3 // 2)
    f = open(video_path, 'rb')
    for i in range(n_frames):
        yuv = np.frombuffer(f.read(width*height*3//2), dtype=np.uint8).reshape((height*3//2, width))
        rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        #rgb = cv2.resize(rgb,(1280,720))
        rgb_frames.append(rgb)
    return  rgb_frames, names





def model_build(batch_shape, model):


	out1 = layers.GlobalAveragePooling2D()(model.output)


	model_final = Model(inputs=model.input,outputs=out1 )

	

	for layer in model_final.layers:
		layer.trainable = False


	return model_final


def extract_feaures(model,list_IDs,features_shape, nb, backbone, batch_size=1, num_patch = 25,overlapping= 0.2):
	videos = DataGenerator(batch_size=batch_size, list_IDs=list_IDs, backbone = backbone, patches = num_patch, overlapping = overlapping, nb =nb )
	name = []
	features_X = np.zeros((nb,num_patch,features_shape))
	features_Y = np.zeros((1))
	i=0 
	for X, Y, ID in tqdm(videos):
		for l in range(nb):
			features = model.predict(X[l,:,:,:,:])

			features_X[l,:,:] = features
			
      

		
		features_Y = Y
		ID = ID.split('.')[0]
		np.save('./features_X/'+ID,features_X)
		np.save('./label/'+ID,features_Y)



if __name__ == '__main__':
 
	parser = argparse.ArgumentParser("Deep_features")

	parser.add_argument('-v',
        '--video_dir',
        default='',
        type=str,
        help='Directory path of videos')



	parser.add_argument('-f',
        '--csv_file',
        default='',
        type=str,
        help='metadata list csv file'
    )

	parser.add_argument('-m',
        '--backbone_model',
        default='resnet50',
        type=str,
        help='backbone_model: resnet50, vgg16, densenet169, inception_v3'
    )

	parser.add_argument('-o',
        '--overlapping', 
        default=0.2, 
        type=float,
        help="overlapping between batches ( between 0 and 1).")


	parser.add_argument('-np',
        '--num_patch',
        default=25,
        type=int,
        help='Number of cropped patches per frames.'
    )
	parser.add_argument('-nf',
        '--num_frames',
        default=30,
        type=int,
        help='Number of cropped frames per video.'
    )



	args = parser.parse_args()


	video_dir = args.video_dir
	video_dir = os.path.expanduser(video_dir)

	if not os.path.exists('./features_X'):
		os.makedirs('./features_X')


	if not os.path.exists('./label'):
		os.makedirs('./label')

	li = prepare_datalist(path_to_csv = args.csv_file , videos_dir= args.video_dir)


	



	num_patch = args.num_patch
	overlap = args.overlapping
	nb = args.num_frames
	backbone = args.backbone_model

	batch_shapes = (num_patch,224,224,3)

	if backbone == 'resnet50':
		model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
		features_shape = 2048
	elif backbone == 'vgg16':
		model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
		features_shape = 512
	elif backbone == 'densenet169':
		model = DenseNet169(weights='imagenet', include_top=False, input_shape=(224,224,3))
		features_shape = 1664
	elif backbone == 'inception_v3':
		model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3))
		features_shape = 2048
	else :
		model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
		features_shape = 2048
		print('Unknown model, using ResNet50... ')
		print('======================================================')






	model_final = model_build(batch_shapes, model)
	print('Starting features extraction process... ')
	print('======================================================')

	extract_feaures(model_final,li, nb = nb, backbone = backbone, features_shape = features_shape,  batch_size=1, num_patch = num_patch,overlapping= overlap)
	

