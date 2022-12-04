import cv2
import numpy as np

file = open("datasets/combined/train", 'r')
image = np.array(file)

image = cv2.resize(image, (100, 100), interpolation=