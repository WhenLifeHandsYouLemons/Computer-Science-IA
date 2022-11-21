from PIL import Image
import pytesseract
import numpy as np
import cv2

# For help on how to get it working: https://stackoverflow.com/questions/46140485/tesseract-installation-in-windows
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

trial_image_1 = "image_01.jpeg"
trial_image_2 = "image_02.png"
trial_image_3 = "image_03.png"

img1 = np.array(Image.open(trial_image_3))

norm_img = np.zeros((img1.shape[0], img1.shape[1]))
img = cv2.normalize(img1, norm_img, 0, 255, cv2.NORM_MINMAX)
img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
img = cv2.GaussianBlur(img, (1, 1), 0)

text = pytesseract.image_to_string(img)
results = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

print(results)

print("\nResult: ", text)
