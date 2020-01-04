
# https://www.youtube.com/watch?v=XNKeayZW4dY
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

print('You have TensorFlow version {}'.format(tf.__version__))

wine_data_path = os.path.abspath('{}/../dataset/wine_data.csv'.format(os.path.dirname(os.path.abspath(__file__))))

print(wine_data_path)

# Load the dataset
data = pd.read_csv(wine_data_path)

# Print 5 rows
data.head()

# remove country not provided
data = data[pd.notnull(data['country'])]
# remove price not provided
data = data[pd.notnull(data['price'])]
# remove first column
data = data.drop(data.columns[0], axis=1)

# Any data that occurs less than 500 times will not be considered
variety_threshold = 500 
value_counts = data['variety'].value_counts()
to_remove = value_counts[value_counts <= variety_threshold].index
data.replace(to_remove, np.nan, inplace=True)
data = data[pd.notnull(data['variety'])]

# Get train data and test data
train_size = int(len(data) * 0.8)
print('Train size:{}'.format(train_size))
print('Test size: {}'.format(len(data) - train_size))

# Train features
description_train = data['description'][:train_size]
variety_train = data['variety'][:train_size]

# Train labels
labels_train = data['price'][:train_size]

# Test features
description_test = data['description'][train_size:]
variety_test = data['variety'][train_size:]

# Test labels
labels_test = data['price'][train_size:]

# Create a tokenizer to preprocess our text descriptions
vocab_size = 3000 # This is a hyperparameter, experiment with different values for your dataset
tokenize = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
tokenize.fit_on_texts(description_train) # only fit on train

# Wide feature 1: sparse bag of words (bow) vocab_size vector
description_bow_train = tokenize.texts_to_matrix(description_train)
description_bow_test = tokenize.texts_to_matrix(description_test)

# Wide feature 2: one-hot vector of variety categories

# Use sklearn utility to convert label string to numbered index
encoder = LabelEncoder()
encoder.fit(variety_train)
variety_train = encoder.transform(variety_train)
variety_test = encoder.transform(variety_test)
num_classes = np.max(variety_train) + 1

# Convert labels to one hot
variety_train = tf.keras.utils.to_categorical(variety_train, num_classes)
variety_test = tf.keras.utils.to_categorical(variety_test, num_classes)

# Define the model with Functional API
bow_inputs = layers.Input(shape=(vocab_size,))
variety_inputs = layers.Input(shape=(num_classes,))
merged_layers = layers.concatenate([bow_inputs, variety_inputs])
merged_layers = layers.Dense(256, activation='relu')(merged_layers)
predictions = layers.Dense(1)(merged_layers)
wide_model = tf.keras.Model(inputs=[bow_inputs, variety_inputs], outputs=predictions)

wide_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
print(wide_model.summary())

# Deep model feature: word embeddings of wine descriptions
train_embed = tokenize.texts_to_sequences(description_train)
test_embed = tokenize.texts_to_sequences(description_test)

max_seq_length = 170
train_embed = tf.keras.preprocessing.sequence.pad_sequences(
	train_embed, maxlen=max_seq_length, padding='post'
)
test_embed = tf.keras.preprocessing.sequence.pad_sequences(
	test_embed, maxlen=max_seq_length, padding='post'
)

# Define our deep model with the Functional API
deep_inputs = layers.Input(shape=(max_seq_length,))
embedding = layers.Embedding(vocab_size, 8, input_length=max_seq_length)(deep_inputs)
embedding = layers.Flatten()(embedding)
embed_out = layers.Dense(1)(embedding)
deep_model = tf.keras.Model(inputs=deep_inputs, outputs=embed_out)
print(deep_model.summary())

deep_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Combine wide and deep model into one model
merged_out = layers.concatenate([wide_model.output, deep_model.output])
merged_out = layers.Dense(1)(merged_out)
combined_model = tf.keras.Model(wide_model.input + [deep_model.input], merged_out)
print(combined_model.summary())

combined_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Run training
combined_model.fit([description_bow_train, variety_train] + [train_embed], np.asarray(labels_train), epochs=10, batch_size=128)

combined_model.evaluate([description_bow_test, variety_test] + [test_embed], np.asarray(labels_test), batch_size=128)

# Generate predictions
predictions = combined_model.predict([description_bow_test, variety_test] + [test_embed])

# Compare predictions with actual values for the first few items in our test dataset
num_predictions = 40
diff = 0

for i in range(num_predictions):
	val = predictions[i]
	print(description_test.iloc[i])
	print('Predicted: {} - Actual: {}'.format(val[0], labels_test.iloc[i]))
	diff += abs(val[0] - labels_test.iloc[i])

print('Average prediction difference: {}'.format(diff / num_predictions))
