import tensorflow as tf
from tf import keras
import numpy as np
training_time = 500
file = open("data.txt", "r+")
current = file.read(1)
currentval = 0
data = ["test", "test"]
while current == "":
	while not current.isspace():
		data[currentval] += current
		current = file.read(1)
	current.read(1)
	currentval += 1
simple_model = keras.Sequential()
for i in range(10):
	simple_model.add(keras.layers.Dense(60, activation = 'relu')
simple_model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
for i in range(training_time):
	simple_model.fit(data, labels, epochs=500, batch_size=32)

