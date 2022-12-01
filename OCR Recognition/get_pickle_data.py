import pickle
import cv2
import numpy as np

# Use this to get data: python extract.py -b 100 -d 2011 2012 2013 -c digits symbols operators -t 1
file = open("CROHME_extractor-master/CROHME_extractor-master/outputs/train/train.pickle", 'rb')
pickle_file = pickle.load(file) # https://www.digitalocean.com/community/tutorials/python-pickle-example
file.close()

print(len(pickle_file[0]["label"]))

image_dimensions = 100

pos = 0
while pos != len(pickle_file):
    if pickle_file[pos]["label"][pos] == 1:
        temp_image = pickle_file[pos]["features"] * 255

        if len(temp_image) != 100*100:
            print("It's not 100*100: ", len(temp_image))

        image = []

        column = 0
        for ii in range(image_dimensions):
            temp_array = []
            for i in range(image_dimensions):
                temp_array.append(temp_image[column])
                column += 1
            image.append(temp_array)

        image = np.array(image)
        image = cv2.resize(image, (100, 100), interpolation=1)

        # cv2.imwrite(f"datasets/CROHME/{pos}.jpg", image)

        cv2.imshow("", image)
        cv2.waitKey(0)

    pos += 1

# The file is as follows:
#   {[features: [image_1, dtype=uint8],
#     label: [len=20 (all are 0, except 1 index which is truth)]],
#    [features: [image_2, dtype=uint8],
#     label: [len=20 (all are 0, except 1 index which is truth)]],
#    ...
#    [features: [image_n, dtype=uint8],
#     label: [len=20 (all are 0, except 1 index which is truth)]]
#   }
# Look at the classes.txt file for the classes used in the file

classes = ["(", ")", "+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=", "[", "/infty", "/int", "/sqrt", "]"]
