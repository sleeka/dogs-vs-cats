# Sean Leeka
# 14 May 2017
# Kaggle data, dogs vs cats
# Taught by Siraj Raval

# import keras	# Machine Learning
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from PIL import Image
import numpy as np 	# Mathematics

# cannot import name load_data
# from parser import load_data 	# data loading

# Image dimensions
img_width, img_height = 150, 150

# Step 1 - Collect Data
train_data_dir = 'train'
# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
	train_data_dir,
	target_size=(img_width, img_height),
	batch_size=16,
	class_mode='binary')

# cannot import name load_data
# training_data = load_data('train')
# below I'll try using the same set for validation
# validation_data = load_data('train')

# Step 2 - Build Model
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

model.compile(loss='binary_crossentropy',
	optimizer='rmsprop',
	metrics=['accuracy'])

# Step 3 - Train Model
model.fit_generator(
	train_generator,
		# Multiplied by ten for bigger dataset (25k images)
	steps_per_epoch=2048,
		# Multiplied by ten for bigger dataset (25k images)
	epochs=2,
	# validation_data=validation_data
	validation_data=train_generator,
		# Multiplied by ten for bigger dataset (25k images)
	validation_steps=832)
model.save_weights('simple_CNN.h5')

# Step 4 - Test Model
img = image.load_img('test/1.jpg', target_size=(224,224))
prediction = model.predict(img)
print prediction

