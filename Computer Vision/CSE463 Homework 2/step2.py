import cv2
import numpy as np


image_path = r'D:\CSE463 Homework 2\image[3].jpg'
output_image_path = r'D:\CSE463 Homework 2\convolved_image_same padding[3].jpg'

kernel = np.array([[0, -1, 0],
                     [-1, 5, -1],
                     [0, -1, 0]])

image = cv2.imread(image_path)
convolved_image = cv2.filter2D(image, -1, kernel)   #basically it is same padding.
cv2.imwrite(output_image_path, convolved_image)

#now i am using padding to make the image (constant,reflect padding,same padding)


output_image_zero_pad = r'D:\CSE463 Homework 2\convolved_image_zero_pad[3].jpg'
output_image_reflect_pad = r'D:\CSE463 Homework 2\convolved_image_reflect_pad[3].jpg'


#zero padding
image_zero_pad = np.pad(image, ((2, 2), (2, 2),(0, 0)), mode='constant')
convolved_image_zero_pad = cv2.filter2D(image_zero_pad, -1, kernel)
cv2.imwrite(output_image_zero_pad, convolved_image_zero_pad)


#reflect padding
image_reflect_pad = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode='reflect')
convolved_image_reflect_pad = cv2.filter2D(image_reflect_pad, -1,kernel)
cv2.imwrite(output_image_reflect_pad, convolved_image_reflect_pad)



