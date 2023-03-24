import cv2
import numpy as np
import face_recognition

imgAlbert_bgr = face_recognition.load_image_file('Images/Albert einstein.jpeg')
imgAlbert_rgb = cv2.cvtColor(imgAlbert_bgr,cv2.COLOR_BGR2RGB)
imgAlbertTest_bgr = face_recognition.load_image_file('Images/AlbertTest.jpeg')
imgAlbertTest_rgb = cv2.cvtColor(imgAlbertTest_bgr,cv2.COLOR_BGR2RGB)


faceloc= face_recognition.face_locations(imgAlbert_rgb)[0]
encodeAlbert = face_recognition.face_encodings(imgAlbert_rgb)[0]
cv2.rectangle(imgAlbert_rgb,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facelocTest= face_recognition.face_locations(imgAlbertTest_rgb)[0]
encodeAlbertTest = face_recognition.face_encodings(imgAlbertTest_rgb)[0]
cv2.rectangle(imgAlbertTest_rgb,(facelocTest[0],facelocTest[2]),(facelocTest[1],facelocTest[3]),(255,0,255),2)

results = face_recognition.compare_faces([encodeAlbert],encodeAlbertTest)
faceDis = face_recognition.face_distance([encodeAlbert],encodeAlbertTest)
print(results,faceDis)
cv2.putText(imgAlbertTest_rgb,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2,cv2.LINE_AA)



cv2.imshow('Albert', imgAlbert_rgb)
cv2.imshow('AlbertTest', imgAlbertTest_rgb)
cv2.waitKey(0)

#imgAlbert =face_recognition.load_image_file('Images/Albert einstein.jpeg')
#imgAlbert = cv2.cvtColor(imgAlbert,cv2.COLOR_BGR2RGB)
##----------Finding face Location for drawing bounding boxes-------
#face = face_recognition.face_locations(imgAlbert_rgb)[0]
#copy = imgAlbert.copy()
##-------------------Drawing the Rectangle-------------------------
#cv2.rectangle(copy, (face[3], face[0]),(face[1], face[2]), (255,0,255), 2)
#cv2.imshow('copy', copy)
#cv2.imshow('Albert',imgAlbert)
#cv2.waitKey(0)

