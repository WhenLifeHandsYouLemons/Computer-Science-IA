import cv2
import numpy as np
import imutils


# Gets the bounding boxes around each number or symbol
def get_bounding_boxes(image_path):
    # Read image from which text needs to be extracted
    image = cv2.imread(image_path)

    image[image > 100] = 255  # type: ignore    # https://stackoverflow.com/questions/19666626/replace-all-elements-of-python-numpy-array-that-are-greater-than-some-value/19666680#19666680

    # Convert the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # Specify structure shape and kernel size.
    # Kernel size increases or decreases the area of the rectangle to be detected.
    # A smaller value like (10, 10) will detect each word instead of a sentence.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 30))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours

# Uses the bounding boxes to crop the image into seperate numbers or symbols
def get_chars(img_path, contours, debug=False):
    # Read image from which text needs to be extracted
    img = cv2.imread(img_path)

    # Creating a copy of image
    im2 = img.copy()

    # Sort the detected contours from left to right ignoring vertical position
    contours = sort_chars(contours)

    # Get the areas of all the detected numbers
    areas = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        areas.append(w*h)

    # Looping through the identified contours
    # Filtering them and a rectangular part is cropped and returned
    area_threshold = 600
    border_size = 25
    chars = []
    i = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if areas[i] > area_threshold:
            # Drawing a rectangle on copied image
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Cropping the text block for giving input to OCR
            cropped = cv2.cvtColor(im2[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

            # Getting total pixels vertically and horizontally
            (tH, tW) = cropped.shape

            # Add a border to make it similar to training data for higher accuracy
            thresh = cv2.copyMakeBorder(  # https://docs.opencv.org/4.x/dc/da3/tutorial_copyMakeBorder.html
                        src=cropped,
                        top=border_size,
                        bottom=border_size,
                        left=border_size,
                        right=border_size,
                        borderType=cv2.BORDER_CONSTANT,
                        value=[255, 255, 255]
                    )

            # if the width is greater than the height, resize along the width dimension
            if tW > tH:
                thresh = imutils.resize(thresh, width=100, height=100)
            # otherwise, resize along the height
            else:
                thresh = imutils.resize(thresh, height=100)

            # Get resized image dimensions and then find out how much width and height needed so our image will be 28x28
            (tH, tW) = thresh.shape # type: ignore
            dX = int(max(0, 100 - tW) / 2)
            dY = int(max(0, 100 - tH) / 2)

            # pad the image and force 28x28 dimensions
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
            padded = cv2.resize(padded, (100, 100))

            chars.append(padded)

            if debug is True:
                cv2.imshow("", img)
                cv2.imshow("", rect)
                cv2.waitKey(0)
        i += 1
    return chars

# Sort the bounding boxes from left to right in the way they appear on screen
def sort_chars(contours):
    # Get all the x-positions of the bounding boxes
    all_x = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        all_x.append(x)

    contours = np.array(contours, dtype=object)

    # Bubble sort the array
    sorted = False
    while sorted is False:
        sorted = True
        for pos in range(len(all_x) - 1):
            if all_x[pos] > all_x[pos + 1]:
                # Use all_x as a basis to sort the contours array
                temp_x = all_x[pos]
                all_x[pos] = all_x[pos + 1]
                all_x[pos + 1] = temp_x
                temp_contour = contours[pos]
                contours[pos] = contours[pos + 1]
                contours[pos + 1] = temp_contour

        for i in range(len(all_x)-1):
            if all_x[i] > all_x[i + 1]:
                sorted = False

    return contours
