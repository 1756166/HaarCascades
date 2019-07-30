import cv2, pickle, numpy as np

#--METHODS--#
def rect_area(tup: tuple): #sorting to get rid of a secondary face - we only want one cascade.
	return tup[2] * tup[3] * 3



#--INSTANCE VARS--#
img_size = 50


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

captures, models = [], []



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
		
		
	cv2.imshow('img', img)

	if len(captures) <= 100:
		if k == ord('s'): # S for Save			
			print('Writing...')
			with open('captures.pickle', 'wb') as f:
				pickle.dump(captures, f)
			print('Done.')
	
	if k == ord('c'): # C for Clear
		print('Cleaning...')
		captures.clear()
		print('Done.')

#--CLOSING PROCESES--#
cv2.destroyAllWindows()
cap.release()