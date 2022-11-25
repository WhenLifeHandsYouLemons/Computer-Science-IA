from tensorflow.keras.datasets import mnist
import numpy as np
import cv2
from sklearn.preprocessing import LabelBinarizer


def load_mnist_dataset():
    # load data from tensorflow framework
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

    # Stacking train data and test data to form single array named data
    data = np.vstack([trainData, testData])

    # Vertical stacking labels of train and test set
    labels = np.hstack([trainLabels, testLabels])

    # return a 2-tuple of the MNIST data and labels
    return (data, labels)


digitsData, digitsLabels = load_mnist_dataset()

data = digitsData
labels = digitsLabels

data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")

# add a channel dimension to every image in the dataset and scale the pixel intensities of the images from [0, 255] down to [0, 1]
data = np.expand_dims(data, axis=-1)
data /= 255.0

le = LabelBinarizer()
labels = le.fit_transform(labels)

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = {}

# loop over all classes and calculate the class weight
for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]
