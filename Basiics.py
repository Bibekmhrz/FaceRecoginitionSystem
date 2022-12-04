import cv2
import numpy as np
import face_recognition

#step1: loading images and convert into RGB
imgBibek = face_recognition.load_image_file("ImageBasic/Bibek.jpg")
imgBibek = cv2.cvtColor(imgBibek, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("ImageBasic/Bibek Test 2.JPG")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

#step2: finding the faces in images and finding their encoding
faceLoc = face_recognition.face_locations(imgBibek)[0]
encodeBibek = face_recognition.face_encodings(imgBibek)[0]
cv2.rectangle(imgBibek, (faceLoc[3],faceLoc[0]), (faceLoc[1],faceLoc[2]), (255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3],faceLocTest[0]), (faceLocTest[1],faceLocTest[2]), (255,0,255),2)

#step3: comparing these faces and finding the distance between them
#encoding for elon and test    ####### we use linear SVM as a backend to find out whether they match or not

results = face_recognition.compare_faces([encodeBibek], encodeTest)
faceDis = face_recognition.face_distance([encodeBibek], encodeTest)     #lower the distance better the match
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX,1, (0,0,255),2)

cv2.imshow("Bibek", imgBibek)
cv2.imshow("Bibek Test", imgTest)
cv2.waitKey(0)