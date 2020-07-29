import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = tf.keras.models.load_model('classify.h5')

class_labels = {0 : 'Mask', 1 : 'No-mask'}

cap = cv2.VideoCapture(0)


while True:
	ret, frame = cap.read()
	if not ret:
		break

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
		roi_gray = gray[y:y+h,x:x+w]
		roi_gray = cv2.resize(roi_gray,(128,128),interpolation=cv2.INTER_AREA)

		roi = roi_gray.astype('float')/255.0
		roi = img_to_array(roi)
		roi = np.expand_dims(roi,axis=0)
		# make a prediction on the ROI, then lookup the class
		preds = classifier.predict(roi)
		maxindex = int(np.argmax(preds))
		if maxindex==1:
			color=(0,0,255)
		else:
			color=(0,255,0)
		cv2.putText(frame, class_labels[maxindex], (x,y), cv2.FONT_HERSHEY_SIMPLEX , 1, color,2)

		
		cv2.imshow('Mask Detector',frame)
		
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()

























