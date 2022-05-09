# import cv2
import cv2 as cv
import numpy as np
import face_recognition

imgElon =face_recognition.load_image_file('imageBasic/Elon musk.jpg')
imgElon =cv.cvtColor(imgElon,cv.COLOR_BGR2RGB)

imgTest =face_recognition.load_image_file('imageBasic/elon musk test.jpg')
imgTest =cv.cvtColor(imgTest,cv.COLOR_BGR2RGB)

faceloc=face_recognition.face_locations(imgElon)[0]
encodeElon =face_recognition.face_encodings(imgElon)[0]
cv.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facelocTest=face_recognition.face_locations(imgTest)[0]
encodeTest =face_recognition.face_encodings(imgTest)[0]
cv.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)

results =face_recognition.compare_faces([encodeElon],encodeTest) #comparing these two faces and their distances
faceDis =face_recognition.face_distance([encodeElon],encodeTest)#distance
print(results,faceDis)
cv.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv.imshow('Elon Musk',imgElon)
cv.imshow('Elon Musk tset',imgTest)
cv.waitKey(0)