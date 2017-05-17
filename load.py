# Sean Leeka
# 14 May 2017
# Kaggle data, dogs vs cats
# Taught by Siraj Raval

# import keras	# Machine Learning
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras import optimizers
from PIL import Image
import numpy as np 	# Mathematics
import cv2, os
# from skimage import io

# cannot import name load_data
# from parser import load_data 	# data loading

# Image dimensions
img_width, img_height = 150, 150

# Step 1 - Build Model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Step 2 - Load Previously Generated Weights
model.load_weights('simple_CNN.h5')

# Step 3 - Load Test Images

# images = cv2.imread('test/*', 1)
# images = np.array('test/')
# images = images.reshape((16, 3, img_width, img_height))

# img = Image.load_img('test/1.jpg', target_size=(224,224))
# test_data_dir = 'test'
# # used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1./255)

# test = []

# for i in range(1,11):
# 	location = 'test/' + str(i) + '.jpg'
# 	test.append(io.imread(location))
# 	x = x.reshape((1, 3 , img_width, img_height))
image = image_utils.load_img('testtest/2.jpg', target_size=(img_width, img_height))
image = image_utils.img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)
pred = model.predict_classes(image)
print pred
exit()
(inID, label) = decode_predictions(pred)[0]
exit()
print inId, label
exit()
test = datagen.flow_from_directory(
	'testtest',
	batch_size=1,
	target_size=(img_width, img_height),
	classes=None,
	shuffle=False)
test_data = model.predict_generator(test, 1)
# test_data_load = test.reshape((1, 150, 150, 3))
np.save(open('test_data.npy', 'w'), test_data)
# test_data_load = np.load(open('test_data.npy', 'rb'))
# test = np.array()
print model.predict_classes(test_data)

# img = Image.open(os.path.join('testtest/2.jpg'))
# resize = image.resize((img_width, img_height), Image.NEAREST)
# resize.load()

# img = np.array(img)
# img = img.reshape((1, 3, 150, 150))
# prediction = model.predict_classes(img)
# print prediction
# test_generator = datagen.flow_from_directory(
# 	'testtest',
# 	target_size=(img_width, img_height),
# 	batch_size=16,
# 	class_mode=None,
# 	shuffle=False)

# print model.predict_classes(test_generator, 10)

