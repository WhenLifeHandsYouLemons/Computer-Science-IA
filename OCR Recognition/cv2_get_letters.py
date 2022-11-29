# Import required packages
import cv2

# Read image from which text needs to be extracted
img = cv2.imread("images/handwritten_test.jpg")

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

# Finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Creating a copy of image
im2 = img.copy()

# Looping through the identified contours
# Then rectangular part is cropped and passed on
organs = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # if (w >= 20) and (h >= 20):
    # Drawing a rectangle on copied image
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    organs.append(rect)

    # Cropping the text block for giving input to OCR
    cropped = im2[y:y + h, x:x + w]

    cv2.imshow("", img)
    cv2.imshow("", rect)
    cv2.waitKey(0)
