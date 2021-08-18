# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:01:30 2021

@author: shahi
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 13:36:19 2021

@author: shahi
"""
import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
thres = 0.45 # Threshold to detect object
nms_threshold = 0.2
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0

# Define the names related to id's
names = ['Jaki', 'Hasib', 'Akash', 'Tithi', 'Naimul', 'Reja', 'Tansu', 'Glass'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
thres = 0.45 # Threshold to detect object
nms_threshold = 0.2
#Define class names for object recognisation

classNames= []
'''Here we have used coco dataset for object recognition'''
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#print(classNames) #for looking into the classes:
'''
We have used configuration and weights provided by openCV 
'''
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'#architecture or the configuration
weightsPath = 'frozen_inference_graph.pb'#weights file

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    
    success,img = cam.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(type(confs[0]))
    #print(confs)

    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    #print(indices)



    ret, img =cam.read()
    #img = cv2.flip(img, -1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=3)
        cv2.putText(img,classNames[classIds[i][0]-1].upper(),(box[0]+10,box[1]+30),
        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)   

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), color =(0,255,0), thickness=3)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "Human"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('Output',img) 
# Press 'ESC' for killing the camera window
    k = cv2.waitKey(10) & 0xff 
    if k == 27:
        break
# Killig Every Contents
print("\n Kill All window, Clean History")
cam.release()
cv2.destroyAllWindows()
