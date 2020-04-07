import tensorflow as tf
import numpy as np


class Model:

    def __init__(self, input_size, output_size):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(input_shape=input_size, units=128, activation=tf.nn.relu),
            tf.keras.layers.Dense(52, activation=tf.nn.relu),
            tf.keras.layers.Dense(output_size, activation=tf.keras.activations.linear)
        ])

        self.model.compile(loss='mean_squared_error',
                           optimizer=tf.keras.optimizers.Adam())

        self.model.summary()

    def train_model(self, train_data):
        input = np.array(train_data[0])
        output = np.array(train_data[1])

        self.model.fit(input, output, epochs=10)

    def predict_model(self, input):
        return self.model.predict(input)
