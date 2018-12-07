import tensorflow as tf
from tensorflow import keras
import numpy as np
#takes a size of the dataset for the given model, and the number of layers, then return a model appropriate for that dataset
def create_model(input_size, layers):
        for i in range(layers):
                model = tf.keras.Sequential()
                model.add(keras.layers.Dense(input_size, activation = 'relu'))
                model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        return model


#loads a model from a file and trains it based on given specs, saves it, and returns it
def train_neural_network_from_file(input, input_size, filename, training_time, checkpoint_epoch):
        model =  tf.keras.models.load_model(filename + ".h5")
        model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        return train_neural_network(input, input_size, filename, model, training_time, checkpoint_epoch)

#takes input dataset and a model and trains it for training_time checkpointing it every checkpoint_epoch epochs, saving to filename then returns it
def train_neural_network(input, input_size, filename, model,  training_time, checkpoint_epoch):
        labels = np.random.random((input_size, input_size))
        for i in range(training_time):
                model.fit(input, labels, epochs=checkpoint_epoch, batch_size=32)
                model.save(filename + "_ver_" + str(i) + ".h5")
                model.save(filename + ".h5")
        return model
#loads a model from filename and then predicts outcome based on input data
def test_neural_network_from_file(input, filename):
        model = tf.keras.models.load_model(filename + ".h5")
        model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        return test_neural_network(input, model)
#uses a predefined model to predict outputs based on input data
def test_neural_network(input, model):
                model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
                return model.predict(input, batch_size=32)
#data = np.array([[1, 1, 1], [12, 13, 14], [55, 56, 57]])
#model = create_model(3, 5)
#model = train_neural_network(data, 3, "models/testmodel333", model, 200, 3)
#model = train_neural_network_from_file(data, 3, "models/testmodel", 20, 2)
#print(test_neural_network_from_file(data, "models/testmodel"))
#result = model.predict(data, batch_size=32)
#print(result)               
