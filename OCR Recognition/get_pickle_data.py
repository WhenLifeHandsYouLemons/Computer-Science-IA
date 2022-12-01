import pickle
import cv2
import numpy as np

file = open("CROHME_extractor-master/CROHME_extractor-master/outputs/train/train.pickle", 'rb')
pickle_file = pickle.load(file) # https://www.digitalocean.com/community/tutorials/python-pickle-example
file.close()

for i in range(4):
    temp_image = pickle_file[i]["features"] * 255

    image = []

    column = 0
    for ii in range(28):
        temp_array = []
        for i in range(28):
            temp_array.append(temp_image[column])
            column += 1
        image.append(temp_array)

    image = np.array(image)

    image = cv2.resize(image, (100, 100), interpolation=1)

    cv2.imshow("", image)
    cv2.waitKey(0)

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
print(len(pickle_file[0]["features"]))

classes = ["(", ")", "+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=", "[", "/infty", "/int", "/sqrt", "]"]
