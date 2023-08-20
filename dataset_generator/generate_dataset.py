#Coding utf-8
"""
dataset作成用プログラム (オフライン)
data augmentationはオフラインなのでメモリを多く消費する可能性あり
→オンラインのバージョンを開発
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

            #print(self.X_image)
                    
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
                #ここに拡張方法を記述する
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.4)

            ])

            #train用データセットを拡張する
            copied_img = copy.deepcopy(self.train_img)
            copied_label = copy.deepcopy(self.train_label)
  
            with tqdm(enumerate(copied_img), 
                      total=len(copied_img), 
                      ncols=70) as augment_list:
                
                augmented = []
                for index, img in augment_list:
                    #1枚の画像を9枚に増やす
                    for i in range(9):
                        augmented_img = augment(img)
                        augmented_label = copied_label[index]
                        augmented.append(augmented_img)
                        self.train_img.append(augmented_img)
                        self.train_label.append(augmented_label)
            
            #拡張した画像を表示する
            plt.figure(figsize=(10, 10))
            for i, img in enumerate(augmented):
                plt.subplot(10, 10, i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(img)
            plt.show()
            
            del copied_img, copied_label   

            self.train_img = np.array(self.train_img)
            self.train_label = np.array(self.train_label)
            self.test_img = np.array(self.test_img)
            self.test_label = np.array(self.test_label)

            #正規化
            self.train_img = self.train_img.astype("float32") / 255
            self.test_img = self.test_img.astype("float32") / 255

        return self.train_img, self.test_img, self.train_label, self.test_label
    

#test
def main(preview=True):
    train_path = "/Users/taikisakai/Desktop/files/"
    extension = "*.jpg"
    class_names = ["fatigue", "ductile", "brittle"]

    generator = CreateDataset(train_path, 
                              extension, 
                              class_names, train_ratio=0.70, test_ratio=0.30)
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
        #iterのしようについては下記を参照
        #https://www.nblog09.com/w/2019/01/12/python-has-next/
        plt.figure(figsize=(10, 10))
        try:
            for i in range(25):
                img, label = next(iterator)
                plt.subplot(5, 5, i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(img)
                plt.xlabel(class_names[int(label)])
            plt.show()

        #画像が25枚以下のときはstopiterationが発生するので例外処理が必要になる
        #25枚以上存在する場合は、例外が発生する前にループが停止する
        except StopIteration:
            plt.show()

    print(dataset)
    print(type(dataset))   
    return dataset, test_img, test_label


if __name__ == "__main__":
    main(preview=True)
