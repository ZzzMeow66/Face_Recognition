import cv2
import dlib
import os
import sys
import random

output_path = './familiar_faces'
size = 64

if not os.path.exists(output_path):
    os.makedirs(output_path)

# change the light and bias for the photograph
def relight(img, light=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    #image = []
    for i in range(0,w):
        for j in range(0,h):
            for k in range(3):
                tmp = int(img[j,i,k]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,k] = tmp
    return img

#using frontal_face_detector from dlib library to extract features of the faces
face_detector = dlib.get_frontal_face_detector()
# turn on the camera to capture photos as input
camera = cv2.VideoCapture(0)

index = 1
while True:
    if (index <= 10000):
        print('Image %s is proccessed' % index)
        # read image from camera
        success, img = camera.read()
        # turn into grey image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # use the detecotor to detect faces
        detects = face_detector(gray_img, 1)

        for i, d in enumerate(detects):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            face = img[x1:y1,x2:y2]
            # adjust light and contrast radio for the pictures
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))

            face = cv2.resize(face, (size,size))

            cv2.imshow('image', face)

            cv2.imwrite(output_path+'/'+str(index)+'.jpg', face)

            index += 1
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        print('All done!')
        break
