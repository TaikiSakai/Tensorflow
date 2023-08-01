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
X_train, X_test = X_train / 255.0, X_test / 255.0


for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i])
    plt.title(str(i))
    plt.tight_layout()

#plt.show()

model = keras.models.Sequential([
    #データを一次元配列に変換
    keras.layers.Flatten(input_shape=(28, 28)),
    #全結合層 512はノード数＆活性化関数
    keras.layers.Dense(512, activation='relu'),
    #データの偏りを減らす
    keras.layers.Dropout(0, 2),
    #全結合層 多クラス分類のときはsoftmaxにする
    keras.layers.Dense(10, activation='softmax')

])

#最適化
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=1000)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')