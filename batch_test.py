
import os, sys
import glob

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array
from keras import datasets, layers, models
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split


train_path = "src"
image_size = 255
color_setting = 1

search_pattern = "*.JPG"
class_names = ["fatigue", "ductile", "brittle", "other"]
class_num = len(class_names)

X_image = []
Y_label = []

for index, name in enumerate(class_names):
    read_data = train_path + "¥¥" + name
    print(read_data)

    for file_path in glob.glob(os.path.join(read_data, search_pattern)):
        print(file_path)
        print("the following class image has been imported: {}".format(index))

        if color_setting == 1:
            img = load_img(file_path, color_mode="grayscale", target_size=(image_size, image_size))

        array = img_to_array(img)
        X_image.append(array)
        Y_label.append(array)

        print(index)
        print("OK1")

print("OK2")

X_image = np.array(X_image)
Y_label = np.array(Y_label)

X_image = X_image.astype("float32") / 255
print(X_image.shape)
