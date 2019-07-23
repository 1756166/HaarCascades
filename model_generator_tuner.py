import tensorflow as tf, numpy as np, os, cv2, random, pickle, matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, MaxPool2D, Dense, Flatten


categories = ['Background', '1']
dataset = []
img_size = 130
model_path = 'C:\\Users\\Krupanidhi\\Desktop\\DownloadedStuff\\Raspi Stuff\\Haar_Cascades\\Models'
	
def load_data():
	
	path = os.path.join(home, 'Desktop', 'DownloadedStuff', 'Raspi Stuff', 'Haar_Cascades', 'Images')
	os.chdir(path)
	
	for category in categories:

		os.chdir(os.path.join(path, category))
		i =  categories.index(category)
		print(i)
	
		with open('captures.pickle') as f:
			pre_set = pickle.load(f)
			for pic in pre_set:
				resized = cv2.resize(pic, (img_size, img_size))
				dataset.append((pic, i))
	
	random.shuffle(dataset)
	imgs, labels = [], []
	
	for (pic, i) in dataset:
		imgs.append(pic)
		labels.append(i)

	images = np.array(images).reshape(-1, img_size, img_size, 1)
	images = images/255.0
	return imgs, labels

def tune_model(imgs: list, labels: list, models: list):		
	
	model = Sequential()
	
	model.add(Conv2D(64, (3, 3), input_shape=dataset.shape[1:]) )
	model.add(Activation('relu'))
	model.add(MaxPool2D((2, 2) ) )
	
	model.add(Conv2D(64, (3, 3) ) )
	model.add(Activation('relu'))
	model.add(MaxPool2D((2, 2) ) )
	
	model.add(Flatten())
	model.add(Dense(64) )
	model.add(Activation('relu'))
	
	model.add(Dense(1)) # Need 1
	model.add(Activation('sigmoid'))
	
	model.compile(optimizer='adam',
	loss='binary_crossentropy',
	metrics=['accuracy'])
	model.fit(imgs, labels, batch_size=20, epochs=3, validation_split=0.15)
	
	models.append(model)