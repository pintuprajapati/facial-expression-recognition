from keras.models import load_model
from time import sleep
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'/haarcascade_frontalface_default.xml')
classifier = load_model(r'/facial_expression_model.h5')

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)  ## it will open the camera and capture the video frame

## If camera is on
while True:
    ## Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) ## Converting into grat scale image, so that expression can be captured easily.
    faces = face_classifier.detectMultiScale(gray, 1.3, 5) ## 3 arguments (input image, scaleFactor, and minNeighbours)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) ## used to draw a rectangle on image. BGR=(255,0,0) which is BLUE color
        roi_gray = gray[y:y+h, x:x+w] ## Region of interest. looks for eyes
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        
        if np.sum([roi_gray]) != 0: ## If anything detected in roi_gray 
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis = 0)
            
        ## Make a prediction on the ROI, then lookup the class
            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()] ## Will select the higest probability class_label
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
       
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            
        cv2.imshow('Facial Expression Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()
        
        
            
        
    