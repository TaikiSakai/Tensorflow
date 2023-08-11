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
        self.Y_label = []
        self.train_img = []
        self.test_img =[]
        self.train_label = []
        self.test_label = []
        self.augmented_img = []
        self.augmented_label = []

        self.img_size = 255
        self.color_setting = 1


    def data_import(self):
        with tqdm(enumerate(self.class_names), 
                  total=len(self.class_names), 
                  ncols=70, 
                  ascii=True) as X_train:
            
            for index, name in X_train:
                read_data = self.src + "//" + name
                for file_path in glob.glob(os.path.join(read_data, self.extenion)):
                    #print(file_path)
                    if self.color_setting == 1:
                        img = load_img(file_path, 
                                       color_mode="grayscale", 
                                       target_size=(self.img_size, self.img_size))
                        
                    #この時点ではarray配列
                    array = img_to_array(img)
                    self.X_image.append(array)
                    self.Y_label.append([int(index)])

                    #print(index)
                    
            print("All images have been imported correctly.")

            return None
        

    def genrate_dataset(self, augmentation=True):
        self.train_img, self.test_img, self.train_label, self.test_label = train_test_split(self.X_image, 
                                                                                            self.Y_label, 
                                                                                            train_size=self.train_ratio, 
                                                                                            test_size=self.test_ratio)
        if augmentation == True:
            print("Executing data augmentation. Wait a moment...")
            augment = tf.keras.Sequential([
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2)
            ])

            #train用データセットを拡張する
            copied_img = copy.deepcopy(self.train_img)
            copied_label = copy.deepcopy(self.train_label)
            
            with tqdm(enumerate(copied_img), 
                      total=len(copied_img), 
                      ncols=70) as augment_list:
                
                for index, img in augment_list:
                    augmented_img = augment(img)
                    augmented_label = copied_label[index]
                    self.train_img.append(augmented_img)
                    self.train_label.append(augmented_label)

            del copied_img, copied_label

            self.train_img = np.array(self.train_img)
            self.train_label = np.array(self.train_label)
            self.test_img = np.array(self.test_img)
            self.test_label = np.array(self.test_label)

            #正規化
            self.train_img = self.train_img.astype("float32") / 255
            self.test_img = self.test_img.astype("float32") / 255

        return self.train_img, self.test_img, self.train_label, self.test_label
    


def main(preview=True):
    train_path = "/Users/taikisakai/Desktop/files/"
    extension = "*.png"
    class_names = ["fatigue", "ductile", "brittle"]

    generator = CreateDataset(train_path, 
                              extension, 
                              class_names, train_ratio=0.60, test_ratio=0.40)
    generator.data_import()
    train_img, test_img, train_label, test_label = generator.genrate_dataset(augmentation=True)

    #train_img, trainlabelをtensor型に変換する
    train_img_ds = tf.data.Dataset.from_tensor_slices(train_img)
    train_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_label, tf.int64))
    dataset = tf.data.Dataset.zip((train_img_ds, train_label_ds)).shuffle(buffer_size=len(train_img))
    print(len(train_img))

    if preview == True:
        iterator = iter(dataset)
        
        #iterのnextでエラー
        #別のバージョンでは動作確認済み
        plt.figure(figsize=(10, 10))
        for i in range(25):
            img, label = next(iterator)
            plt.subplot(5, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img)
            plt.xlabel(class_names[int(label)])
        plt.show()
        
    return dataset, test_img, test_label


if __name__ == "__main__":
    main()
