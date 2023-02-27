import cv2
import numpy as np

# Gets the contours of the individual characters in the image
def get_contours(IMG_PATH):
    # Read image from which text needs to be extracted
    image = cv2.imread(IMG_PATH)

    # Make the edges sharper by removing the gradient between the writing and the background
    image[image > 75] = 255    # https://stackoverflow.com/a/19666680

    # Convert the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # Specify structure shape and kernel size
    # Kernel size increases or decreases the area of the rectangle to be detected
    # A smaller value like (10, 10) will detect each word instead of a sentence
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))

    # Applying dilation on the threshold image
    dilation_image = cv2.dilate(thresh1, rect_kernel, iterations = 1)


    # Finding contours
    contours, hierarchy = cv2.findContours(dilation_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Return an array of contours
    return contours

# Sort the bounding boxes (contours) from left to right in the way they appear on screen
def sort_chars(CONTOURS):
    # Get all the x positions of the bounding boxes
    all_x = []
    for contour in CONTOURS:
        x, y, w, h = cv2.boundingRect(contour)
        all_x.append(x)

    # Convert to a numpy array
    contours = np.array(CONTOURS, dtype=object)

    # Using bubble sort
    sorted = False
    while sorted is False:
        sorted = True
        # Go through each item in the all_x array
        for pos in range(len(all_x) - 1):
            # Check whether the current item is larger than the next one
            if all_x[pos] > all_x[pos + 1]:
                # Use all_x as a basis to sort the contours array
                # Switch the items in the all_x array
                temp_x = all_x[pos]
                all_x[pos] = all_x[pos + 1]
                all_x[pos + 1] = temp_x
                # Switch the items in the contours array
                temp_contour = contours[pos]
                contours[pos] = contours[pos + 1]
                contours[pos + 1] = temp_contour

        # Check whether the array is sorted
        for i in range(len(all_x)-1):
            if all_x[i] > all_x[i + 1]:
                sorted = False

    # Return sorted array
    return contours

# Get each character from the contours and return the split images
def get_chars(IMG_PATH, CONTOURS, AREA_THRESH, BORDER_SIZE, DEBUG = False):
    # Read image from which text needs to be extracted
    img = cv2.imread(IMG_PATH)
    # Creating a copy of image
    img_c = img.copy()

    # Get the areas of all the detected numbers
    areas = []
    for cnt in CONTOURS:
        x, y, w, h = cv2.boundingRect(cnt)
        areas.append(w*h)

    # Looping through the identified contours
    # Filtering them and a rectangular part is cropped and returned
    area_threshold = AREA_THRESH
    border_size = BORDER_SIZE
    chars = []
    i = 0
    for cnt in CONTOURS:
        x, y, w, h = cv2.boundingRect(cnt)

        if areas[i] > area_threshold:
            # Drawing a rectangle on copied image
            rect = cv2.rectangle(img_c, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Cropping the text block for giving input to OCR
            cropped = cv2.cvtColor(img_c[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

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

            # Resize image to (100, 100) if needed  # https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py
            inter = cv2.INTER_AREA
            dim = None
            (h, w) = thresh.shape[:2]
            width, height = 100, 100
            # if the width is greater than the height, resize along the width dimension
            if tW > tH:
                r = width / float(w)
                dim = (width, int(h * r))
                thresh = cv2.resize(thresh, dim, interpolation=inter)
            # otherwise, resize along the height
            else:
                r = height / float(h)
                dim = (int(w * r), height)
                thresh = cv2.resize(thresh, dim, interpolation=inter)

            # Get resized image dimensions and then find out how much width and height needed so our image will be 28x28
            (tH, tW) = thresh.shape # type: ignore
            dX = int(max(0, 100 - tW) / 2)
            dY = int(max(0, 100 - tH) / 2)

            # pad the image and force 28x28 dimensions
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
            padded = cv2.resize(padded, (100, 100))

            padded[padded > 75] = 255

            chars.append(padded)

            if DEBUG is True:
                cv2.imshow("", img)
                # cv2.imshow("", padded)
                cv2.imshow("", rect)
                cv2.waitKey(0)
        i += 1
    return chars
