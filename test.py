import tensorflow as tf
from tensorflow import keras
testmodel = tf.keras.Sequential();
testmodel.add(keras.layers.Dense(64, activation = 'relu'))
#testmodel.add(keras.layers.Dense(10, activation = 'softmax'))
testmodel.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 64))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 64))

testmodel.fit(data, labels, epochs=10, batch_size=32)
          #validation_data=(val_data, val_labels))
