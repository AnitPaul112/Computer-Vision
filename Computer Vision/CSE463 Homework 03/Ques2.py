#Part 1: The Cat’s Trickery
import cv2
import numpy as np

image_path = r'D:\CSE463 Homework 03\Cat.jpg'
cat_image = cv2.imread(image_path)


# Shrinking Spell
shrunken_cat = cv2.pyrDown(cat_image)
cv2.imwrite(r'D:\CSE463 Homework 03\Cat_Shrunken.jpg', shrunken_cat)

# Teleportation Act
num_rows, num_cols = cat_image.shape[:2]
translation_matrix = np.float32([[1, 0, 500], [0, 1, 500]])  # Shift right by 50 and down by 30
translated_cat = cv2.warpAffine(cat_image, translation_matrix, (num_cols, num_rows))
cv2.imwrite(r'D:\CSE463 Homework 03\Cat_Teleported.jpg', translated_cat)

# Twisting Tail Move
rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), 45, 1)  # Rotate by 45 degrees
rotated_cat = cv2.warpAffine(cat_image, rotation_matrix, (num_cols, num_rows))
cv2.imwrite(r'D:\CSE463 Homework 03\Cat_Rotated.jpg', rotated_cat)

# Sunbeam Glow
bright_cat = cv2.convertScaleAbs(cat_image, alpha=1.5, beta=50)  # Increase brightness
cv2.imwrite(r'D:\CSE463 Homework 03\Cat_Brightened.jpg', bright_cat)




#Part2: Deploying SIFT the Cat Tracker





cat_gray = cv2.cvtColor(cat_image, cv2.COLOR_BGR2GRAY)

shrunken_cat_gray = cv2.cvtColor(shrunken_cat, cv2.COLOR_BGR2GRAY)

translated_cat_gray = cv2.cvtColor(translated_cat, cv2.COLOR_BGR2GRAY)

rotated_cat_gray = cv2.cvtColor(rotated_cat, cv2.COLOR_BGR2GRAY)

bright_cat_gray = cv2.cvtColor(bright_cat, cv2.COLOR_BGR2GRAY)



sift = cv2.SIFT_create()

#Detect keypoints and descriptors
def detect_keypoints_and_descriptors(image_gray):
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)
    return keypoints, descriptors


original_keypoints, original_descriptors = detect_keypoints_and_descriptors(cat_gray)
shrunken_keypoints, shrunken_descriptors = detect_keypoints_and_descriptors(shrunken_cat_gray)
translated_keypoints, translated_descriptors = detect_keypoints_and_descriptors(translated_cat_gray)
rotated_keypoints, rotated_descriptors = detect_keypoints_and_descriptors(rotated_cat_gray)
bright_keypoints, bright_descriptors = detect_keypoints_and_descriptors(bright_cat_gray)

#Keypoint matching 
def match_and_save_results(original_image, original_keypoints, original_descriptors, transformed_image, transformed_keypoints, transformed_descriptors, output_path):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.match(original_descriptors, transformed_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance
    
    

 

# Perform matching for each transformed image
match_and_save_results(cat_image, original_keypoints, original_descriptors, shrunken_cat, shrunken_keypoints, shrunken_descriptors, r'D:\CSE463 Homework 03\Cat_Shrunken_Matches.jpg')
match_and_save_results(cat_image, original_keypoints, original_descriptors, translated_cat, translated_keypoints, translated_descriptors, r'D:\CSE463 Homework 03\Cat_Teleported_Matches.jpg')
match_and_save_results(cat_image, original_keypoints, original_descriptors, rotated_cat, rotated_keypoints, rotated_descriptors, r'D:\CSE463 Homework 03\Cat_Rotated_Matches.jpg')
match_and_save_results(cat_image, original_keypoints, original_descriptors, bright_cat, bright_keypoints, bright_descriptors, r'D:\CSE463 Homework 03\Cat_Brightened_Matches.jpg')




#Cat and dog 
dog_image_path = r'D:\CSE463 Homework 03\Dog.jpg'


dog_image = cv2.imread(dog_image_path)
dog_gray = cv2.cvtColor(dog_image, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
cat_keypoints, cat_descriptors = sift.detectAndCompute(cat_gray, None)
dog_keypoints, dog_descriptors = sift.detectAndCompute(dog_gray, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


matches = bf.match(cat_descriptors, dog_descriptors)
matches = sorted(matches, key=lambda x: x.distance)
result_image = cv2.drawMatches(cat_image, cat_keypoints, dog_image, dog_keypoints, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

output_path = r'D:\CSE463 Homework 03\Cat_Dog_Matches.jpg'
cv2.imwrite(output_path, result_image)


'''
SIFT does not inherently understand the semantic differences between a cat and a dog, it focuses solely on comparing local patterns in the images. 
If the local textures or shapes share similarities, SIFT might still produce matches, even if the images are fundamentally different. 
This highlights its strength in effectively matching regions with similar patterns, even across transformations like rotation or scale changes. 
However, its limitation lies in its inability to evaluate the overall structure or meaning of the image, leading to potential mismatches where local features of a cat (e.g., fur patterns) might be mistaken for parts of a dog.
For meaningful discrimination, additional context-aware algorithms (e.g., deep learning-based object recognition) would be more effective.
'''