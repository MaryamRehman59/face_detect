
import cv2 as cv
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

img = cv.imread('images/bluegirl.jpg')
cv.imshow('girl', img)

gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gery', gray)


#contrast control filter
new_image = np.zeros(img.shape, img.dtype)
alpha = 1.0 
beta = 0 
new_image = cv.convertScaleAbs(img, alpha=1.1, beta=10)

#bitertal filter
biletral = cv.bilateralFilter(new_image, 5, 80, 80)



#BGR to RBG
rgb = cv.cvtColor(biletral, cv.COLOR_BGR2RGB)

#Blue, Green, Red Channels
b, g, r = cv.split(biletral)
blank = np.zeros(new_image.shape[:2], dtype='uint8' )
blue= cv.merge([b, blank, blank])
green= cv.merge([blank, g, blank])
red= cv.merge([blank, blank, r])

#Color Histogram of image
plt.figure()
plt.title('color Histogram')
plt.xlabel('Bins')
plt.ylabel('No of Pixels')

color= ('b', 'g', 'r')
for i, col in enumerate(color):
    hist= cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])  


#face detection
image= img
haar_cascade = cv.CascadeClassifier('haar_face.xml')
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=40)
print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(image, (x,y), (x+w,y+h), (150,100,200), thickness=2)
cv.imshow('Detected Faces', image)


if len(faces_rect) >= 0:
    cv.imshow('biletral', biletral)
    cv.imshow('New Image', new_image)
    cv.imshow('BGR_image', rgb)
    cv.imshow('blue', blue)
    cv.imshow('green', green)
    cv.imshow('red', red)
    plt.show()
else:
    print('no faces found')
    exit(0)

cv.waitKey(0)
