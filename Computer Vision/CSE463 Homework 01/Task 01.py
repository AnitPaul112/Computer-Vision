import numpy as np
import matplotlib.pyplot as plt

Letter_A = np.array([[0, 0, 0, 0, 0],
                     [0, 255, 255, 255, 0],
                     [0, 0, 0, 0, 0],
                     [0, 255, 255, 255, 0],
                     [0, 255, 255, 255, 0]])

Number_5 = np.array([[0, 0, 0, 0, 0],
                     [0, 255, 255, 255, 255],
                     [0, 0, 0, 0, 0],
                     [255, 255, 255, 255, 0],
                     [0, 0, 0, 0, 0]])

Number_8 = np.array([[0, 0, 0, 0, 0],
                     [0, 255, 255, 255, 0],
                     [0, 0, 0, 0, 0],
                     [0, 255, 255, 255, 0],
                     [0, 0, 0, 0, 0]])


A58 = np.hstack([Letter_A, np.full((5, 3), 255), Number_5, np.full((5, 3), 255), Number_8])


plt.imshow(A58, cmap='gray')
plt.axis('off')  
plt.show()
