import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('D:\plot_example.png',cv2.IMREAD_COLOR)
'''
cv2.imshow("image", img) #BGR format

cv2.waitKey(0)
cv2.destroyAllWindows()

print(img.shape)
#converting BGR to RGB color format
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(RGB_img) #RGB format
plt.waitforbuttonpress()
plt.close('all')'''

#Grayscale

img = cv2.imread('D:\plot_example.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Image Resizing
'''img = cv2.imread('D:\plot_example.png', 1)
half = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
bigger = cv2.resize(img, (1050,1610))

stretch_near = cv2.resize(img, (780,540), interpolation = cv2.INTER_LINEAR)

Titles = ["original", 'half', 'bigger', 'interpolation nearest']
images = [img, half, bigger, stretch_near]
count = 4
for i in range(count):
    plt.subplot(2, 2, i+1)
    plt.title(Titles[i])
    plt.imshow(images[i])
plt.show()'''

#Eroding an image
'''
img = cv2.imread('D:/plot_example.png')

#Creating kernel
kernel = np.random.randint(0,1,(5,5),dtype="uint8")

#Using cv2.erode() method
image = cv2.erode(img, kernel)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
#Image Blurring
cv2.imshow("Original Image", img)
cv2.waitKey(0)

#gaussian blur
Gaussian = cv2.GaussianBlur(img, (7,7), 0)
cv2.imshow("Gaussian Blur", Gaussian)
cv2.waitKey(0)
#Median Blur
median = cv2.medianBlur(img, 5)
cv2.imshow('Median Blurring', median)
cv2.waitKey(0)
#Bilateral Blur
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imshow('Bilateral Blur', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''

'''
img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value=0)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
gray_img = cv2.imread('D:/plot_example.png', 0)#0 means graycode
cv2.imshow("grayscale", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
#Using pixel manipulation
#Obtain dimensions using shape method
(row, col) = img.shape[0:2]

#Take the average of pixel values of the BGR Channels
#to convert the colored image to grayscale image
for i in range(row):
    for j in range(col):
        #Find the average of BGR pixel values
        img[i,j] = sum(img[i,j])*0.33
cv2.imshow('grayscale image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

############################################
#Point to be noted even though all grayscale methods must give same output last one is a bit different
##Y though??