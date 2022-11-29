import pickle

file = open("CROHME_extractor-master/CROHME_extractor-master/outputs/train/train.pickle", 'rb')
pickle_file = pickle.load(file) # https://www.digitalocean.com/community/tutorials/python-pickle-example
file.close()

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
print(len(pickle_file))
