# -*- codeing: utf-8 -*-
import sys
import os
import cv2
import dlib

input_path = './input_img'
output_path = './unfamiliar_faces'
size = 64

if not os.path.exists(output_path):
    os.makedirs(output_path)

#using frontal_face_detector from dlib library to extract features of the faces
face_detector = dlib.get_frontal_face_detector()

index = 1
for (path, dirnames, filenames) in os.walk(input_path):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Image %s is proceessed' % index)
            img_path = path+'/'+filename
            # read image
            img = cv2.imread(img_path)
            # turn into grey picture
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detects = face_detector(gray_img, 1)

            for i, d in enumerate(detects):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                # img[y:y+h,x:x+w]
                face = img[x1:y1,x2:y2]
                # adjust size of the image
                face = cv2.resize(face, (size,size))
                cv2.imshow('image',face)
                # save image
                cv2.imwrite(output_path+'/'+str(index)+'.jpg', face)
                index += 1

            key = cv2.waitKey(30) & 0xff
            if key == 27:
                sys.exit(0)
