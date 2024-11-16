import cv2

image_path = r'D:\CSE463 Homework 2\image[7].jpg' 
clear_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  


equalized_image = cv2.equalizeHist(clear_image)
output_equalized_image = r'D:\CSE463 Homework 2\equalized_image[7].jpg'

cv2.imwrite(output_equalized_image, equalized_image)

#Histogram equalization improves contrast by evenly spreading out pixel intensity values, making darker or lighter regions more visible.




#Applying he 3 times


import numpy as np

image_path = r'D:\CSE463 Homework 2\image[7].jpg'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


equalized_once = cv2.equalizeHist(original_image)
equalized_twice = cv2.equalizeHist(equalized_once)
equalized_thrice = cv2.equalizeHist(equalized_twice)

combined_image = np.hstack((equalized_once, equalized_twice, equalized_thrice))


output_path = r'D:\CSE463 Homework 2\equalized_image_combined.jpg'
cv2.imwrite(output_path, combined_image)

#Repeated applications of histogram equalization lead to diminishing returns as the histogram becomes uniform after the first pass.
# Further iterations can sometimes introduce artifacts or make the image appear unnaturally enhanced.