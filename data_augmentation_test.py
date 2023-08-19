import os, sys
import copy
import glob

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.utils import load_img, img_to_array
from keras import datasets, layers
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(rescale=1. / 255)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=10)


