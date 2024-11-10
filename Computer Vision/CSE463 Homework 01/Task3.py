import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image_paths = [
    r'D:\CSE463 Homework 01\Urban Images\urban 1.jpeg',
    r'D:\CSE463 Homework 01\Urban Images\urban 2.jpeg',
    r'D:\CSE463 Homework 01\Urban Images\urban 3.jpeg',
    r'D:\CSE463 Homework 01\Urban Images\urban 4.jpeg',
    r'D:\CSE463 Homework 01\Urban Images\urban 5.jpeg'
]

output_folder = r'D:\CSE463 Homework 01\Urban Images\Output'
for idx, path in enumerate(image_paths):
    img = cv2.imread(path)
    mean = 0
    std_dev = 25 
    noise = np.random.normal(mean, std_dev, img.shape).astype(np.uint8)
    noisy_image = cv2.add(img, noise)
  
    cv2.imwrite(os.path.join(output_folder, f'original_image_{idx + 1}.jpeg'), img)
    cv2.imwrite(os.path.join(output_folder, f'noisy_image_{idx + 1}.jpeg'), noisy_image)


    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f'Original Image {idx + 1}')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'Noisy Image {idx + 1}')
    plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

    #histogram
    plt.figure(figsize=(5, 5))
    plt.hist(noisy_image.ravel(), bins=256, color='red', alpha=0.5, label='Noisy Image')
    plt.title(f'Histogram of Noisy Image {idx + 1}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()



#For blending the images

base_image = cv2.imread(image_paths[0])
height, width, channels = base_image.shape
accum_image = np.zeros((height, width, channels), dtype=np.float32)

for path in image_paths:
    img = cv2.imread(path)
    if img.shape != (height, width, channels):
        img = cv2.resize(img, (width, height))
    accum_image += img.astype(np.float32)


blended_image = (accum_image / len(image_paths)).astype(np.uint8)


cv2.imwrite("blended_image.jpg", blended_image)
plt.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))