import os, sys
import glob

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

batch_size = 128
num_class = 10
epoch = 20

(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()# numpy ndarray

#モデルの正規化
train_images, test_images = X_train / 255.0, X_test / 255.0

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(train_images[i])
    plt.title(str(i))
    plt.tight_layout()

plt.show()


    