import cv2
import numpy as np
import matplotlib.pyplot as plt


mean = 0
std_dev = 50  

img = cv2.imread(r'D:\CSE463 Homework 2\image [4].jpg', cv2.IMREAD_COLOR)

# Generate Gaussian noise with the same shape as the image
noise = np.random.normal(mean, std_dev, img.shape).astype(np.float32)

output = cv2.add(img.astype(np.float32), noise)
output = np.clip(output, 0, 255).astype(np.uint8)  


output_path = r'D:\CSE463 Homework 2\image_noisy[4].jpg'
cv2.imwrite(output_path, output)


filtered_output = cv2.blur(output, (5, 5))
filtered_output_path = r'D:\CSE463 Homework 2\image_filtered_noisy[4].jpg'
cv2.imwrite(filtered_output_path, filtered_output)

#Applying a 5x5 average filter reduces Gaussian noise but results in a slightly blurred appearance. 
# Fine details are smoothed along with the noise.


# Apply Gaussian blur with different sigma values
blurred_sigma1 = cv2.GaussianBlur(output, (5, 5), 1)   # Sigma = 1
blurred_sigma2 = cv2.GaussianBlur(output, (5, 5), 2)   # Sigma = 2
blurred_sigma3 = cv2.GaussianBlur(output, (5, 5), 3)   # Sigma = 3


cv2.imwrite(r'D:\CSE463 Homework 2\image_blurred_sigma1.jpg', blurred_sigma1)
cv2.imwrite(r'D:\CSE463 Homework 2\image_blurred_sigma2.jpg', blurred_sigma2)
cv2.imwrite(r'D:\CSE463 Homework 2\image_blurred_sigma3.jpg', blurred_sigma3)



# Gaussian blur provides more controlled smoothing. 
# Increasing the standard deviation (σ) results in stronger smoothing, but excessive values may over-blur the image and remove details.



