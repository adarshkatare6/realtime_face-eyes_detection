
import cv2 as cv
import numpy as np

def detect(img):
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    haar_cascade=cv.CascadeClassifier('haar_face.xml')
    haar_cascade_eye=cv.CascadeClassifier('haar_eye.xml')
    if haar_cascade.empty():
        print("Error loading face cascade file")
    elif haar_cascade_eye.empty():
        print("Error loading eye cascade file")
    else:
        face_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
        eye_rect=haar_cascade_eye.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
        for (x,y,w,h) in face_rect:
            cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        for (m,n,o,p) in eye_rect:
            cv.rectangle(img,(m,n),(m+o,n+p),(255,0,0),2)
        # print("no of faces detected=",len(face_rect))
        # print("no of eyes detected=",len(eye_rect))
        cv.putText(img, f"Faces: {len(face_rect)}, Eyes: {len(eye_rect)}", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        img=cv.resize(img,(640,480))
    return img

#main
cap=cv.VideoCapture(0)
while True:
    true,frame=cap.read()
    if not true:
        print("failed to caapture")
        break
    flip=cv.flip(frame,1)
    image=detect(flip)
    cv.imshow("vedio",image)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
cap.release()
cv.destroyAllWindows()