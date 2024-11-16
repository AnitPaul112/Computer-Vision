import cv2
import numpy as np


image_path = r'D:\CSE463 Homework 2\image[1].jpg'
output_image_path = r'D:\CSE463 Homework 2\convolved_image[1].jpg'

image = cv2.imread(image_path)
#a basic 3*3 identity kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

convolved_image = cv2.filter2D(image, -1, kernel)
cv2.imwrite(output_image_path, convolved_image)


#custom kernel for sharpening

kernel2 = np.array([[0, -1, 0],
                     [-1, 5, -1],
                     [0, -1, 0]])

image_path2 = r'D:\CSE463 Homework 2\image[2].jpg'
output_image_path2 = r'D:\CSE463 Homework 2\convolved_image[2].jpg'

image = cv2.imread(image_path2)
convolved_image2 = cv2.filter2D(image, -1, kernel2)
cv2.imwrite(output_image_path2, convolved_image2)