import cv2
import numpy as np

image_path = r'D:\CSE463 Homework 2\image[5].jpg'

clear_image = cv2.imread(image_path, cv2.IMREAD_COLOR)  

# Apply Laplacian filter
laplacian_filtered_image = cv2.Laplacian(clear_image, cv2.CV_64F) 
laplacian_filtered_image = cv2.convertScaleAbs(laplacian_filtered_image)  


output_path = r'D:\CSE463 Homework 2\image_edgecolor[5].jpg'
cv2.imwrite(output_path, laplacian_filtered_image)


#greyscale image
gray_image = cv2.cvtColor(clear_image, cv2.COLOR_BGR2GRAY)
# Apply Laplacian filter
laplacian_filtered_image = cv2.Laplacian(gray_image, cv2.CV_64F)

output_path = r'D:\CSE463 Homework 2\image_edgegray[5].jpg'
cv2.imwrite(output_path, laplacian_filtered_image)

#he Laplacian filter highlights edges by detecting changes in intensity. 
# Applied to the grayscale image, it provides a clean edge map without color artifacts.





image_path_2 = r'D:\CSE463 Homework 2\image[6].jpg'
clear_image_2 = cv2.imread(image_path_2, cv2.IMREAD_COLOR)

horizontal_kernel = np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]])

vertical_kernel = np.array([[-1, -1, -1],
                            [0,  0,  0],
                            [1,  1,  1]])

# Apply the horizontal kernel to detect horizontal edges
horizontal_edges = cv2.filter2D(clear_image_2, cv2.CV_64F, horizontal_kernel)
horizontal_edges = cv2.convertScaleAbs(horizontal_edges)

# Apply the vertical kernel to detect vertical edges
vertical_edges = cv2.filter2D(clear_image_2, cv2.CV_64F, vertical_kernel)
vertical_edges = cv2.convertScaleAbs(vertical_edges)


output_horizontal_path = r'D:\CSE463 Homework 2\horizontal_edges[5].jpg'
output_vertical_path = r'D:\CSE463 Homework 2\vertical_edges[5].jpg'
cv2.imwrite(output_horizontal_path, horizontal_edges)
cv2.imwrite(output_vertical_path, vertical_edges)

#The horizontal kernel captures edges along the vertical direction, while the vertical kernel captures edges along the horizontal direction.
#Combined, they provide comprehensive edge detection