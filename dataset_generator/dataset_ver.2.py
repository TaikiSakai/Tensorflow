#Coding utf-8
"""
dataset作成用プログラム
"""

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
from tqdm import tqdm


class CreateDataset:

    def __init__(self, 
                 src, 
                 extension, 
                 class_names, 
                 train_ratio, 
                 test_ratio):
        
        self.src = src
        self.extenion = extension
        self.class_names = class_names
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

        self.X_image = [] 
        self.Y_image = []
        self.train_img = []
        self.test_img =[]
        self.train_label = []
        self.test_label = []
        self.augmented_img = []
        self.augmented_label = []

        self.img_size = 255
        self.color_setting = 1


    def data_import(self):
        with tqdm(enumerate(class_names), 
                  total=len(self.class_names), 
                  ncols=70, 
                  ascii=True) as X_train:
            
            for index, img in X_train:
                read_data = self.src + "//" + name
                for file_path in glob.glob(os.path.join(read_data, self.extenion)):

                    sys.exit(0)