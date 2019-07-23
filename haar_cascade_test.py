import cv2, os, pathlib, datetime as dt, pickle, model_generator_tuner as mgf, numpy as np, tensorflow as tf, time

#--METHODS--#
def rect_area(tup: tuple): #sorting to get rid of a secondary face - we only want one cascade.
	return tup[2] * tup[3] * 3

def load_init_model():
	os.chdir('C:\\Users\\Krupanidhi\\Desktop\\Videos (in case YouTube becomes a thing)\\Science Project\\Code\\models')
	model = tf.keras.models.load_model('science_project.model')
	return model

#--INSTANCE VARS--#
img_size = 50

home = str(pathlib.Path.home())

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

captures, models = [], []
models.append(load_init_model())

path = os.path.join(home, 'Desktop', 'DownloadedStuff', 'Raspi Stuff', 'Haar_Cascades', 'Images')
os.chdir(path)

cap = cv2.VideoCapture(0)

#--MAIN BODY--#	
while 1:

	k = cv2.waitKey(5) & 255
	if k == 27:
		break
	
	_, img = cap.read()
	
	faces = face_cascade.detectMultiScale(img, 1.3, 5)	
	for x, y, w, h in faces:
		cv2.rectangle(img, (x - 20 , y - 20), (x + w + 20, y + h + 20), (255, 0, 0))

	if type(faces) is np.ndarray:
	
		faces = list(faces)
		faces.sort(key = rect_area, reverse=True)
		face = faces[0] # Isolate the largest face
		
		x, y, w, h = face
		roi = img[y-20:y+h+20, x-20:x+w+20]
		
		try:
			gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		except Exception as e:
			pass
		
		predict_img = cv2.resize(roi, (img_size, img_size)).reshape(-1, img_size, img_size, 1)
		cv2.imshow('largest', roi)
		
		for model in models:
			print('We Heresss')
			if int(model.predict([predict_img])[0]) != 1: # If the image isn't recognized
				print('I don{}t even know who you are.'.format('\''))
				if len(captures) < 100:
					captures.append(gray)
					print('Creating List...')
					time.sleep(1)
				else:
					print('Too Full...', 'Would you like to train your image?') # On Raspi, this would be replaced by an LED or a speaker of some sort.

	cv2.imshow('img', img)

	if k == ord('s'): # S for Save			
		print('Writing...')
		with open('captures.pickle', 'wb') as f:
			pickle.dump(captures, f)
		print('Done.')
	
	if k == ord('c'): # C for Clear
		print('Cleaning...')
		captures.clear()
		print('Done.')

	if k == ord('t'): # T for Train
		print('Generating Model...')
		imgs, labels = mgf.load_data()
		imgs = np.array(imgs).reshape(-1, img_size, img_size, 1)
		imgs = imgs/255.0
		mgf.tune_model(imgs, labels, models)
		print('Done.')
		captures.clear()

#--CLOSING PROCESES--#
cv2.destroyAllWindows()
cap.release()