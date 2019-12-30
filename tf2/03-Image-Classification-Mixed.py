from __future__ import absolute_import, division, print_function, unicode_literals

# Load tensorflow

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout

# Load the data for training and testing

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Create the model

model = tf.keras.models.Sequential([
    Conv2D(32, 3, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# Compile the module

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model

model.fit(x_train, y_train, epochs=5)

# Is it good?

model.evaluate(x_test, y_test, verbose=2)

