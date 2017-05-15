# Sean Leeka
# 14 May 2017
# Kaggle data, dogs vs cats
# Taught by Siraj Raval

import keras	# Machine Learning
import numpy as np 	# Mathematics
from parses import load_data 	# data loading

# Step 1 - Collect Data
training_data = load_data('data/training')
validation_data = load_data('data/validation')

# Step 2 - Build Model
model = Sequential()
model.add(Convolution2D(32,3,3 input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,3,3))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
	optimizer='rmsprop',
	metrics=['accuracy'])

# Step 3 - Train Model
model.fit_generator(
	training_data,
	samples_per_epoch=2048,
	nb_epoch=2048,
	validation_data=validation_data
	nb_val_samples=832)
model.save_weights('models/simple_CNN.h5')

# Step 4 - Test Model
img = image.load_img('test/img001.jpg', target_size=(224,224))
prediction = model.predict(img)
print prediction

