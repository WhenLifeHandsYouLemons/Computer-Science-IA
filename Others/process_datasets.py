import os
from PIL import Image

PATH = "datasets/sagyamthapa/dataset/"

classes = os.listdir(f"{PATH}")

dimensions = (100, 100)
for current_class in classes:
    files = os.listdir(f"{PATH}{current_class}/")
    for file in files:
        image = Image.open(f"{PATH}{current_class}/{file}")
        image = image.resize(dimensions)    # https://auth0.com/blog/image-processing-in-python-with-pillow/
        image.save(f"{PATH}{current_class}/{file}")

print("Completed successfully")
