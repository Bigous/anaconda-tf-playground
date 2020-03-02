import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

from tqdm import tqdm
from urllib.request import urlretrieve

import tensorflow as tf

import matplotlib.pyplot as plt
# %matplotlib inline

import os
import re
import random as rn
import numpy as np

os.environ['PYTHONHASHSEED'] = '0'
RAMDON_SEED = 3939
np.random.seed(RAMDON_SEED)
rn.seed(RAMDON_SEED)
tf.compat.v1.set_random_seed(RAMDON_SEED)

DATA_URL = "http://www.gutenberg.org/cache/epub/1041/pg1041.txt"
DATA_FILENAME = "sonnets.txt"

SEQ_LENGTH = 100
FEATURE_NUM = 1


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


with DLProgress(unit="B", unit_scale=True, miniters=1, desc="Shakespeare's Sonnets") as pbar:
    urlretrieve(DATA_URL, DATA_FILENAME, pbar.hook)


with open(DATA_FILENAME, "r") as file:
    data = file.read()

start_index = 740
end_index = re.search("Love's fire heats water, water cools not love.", data)
end_index = end_index.end()

data_cleaned = data[start_index:end_index]
# print(data_cleaned)

removal_list = ["\xbb", "\xbf", "\xef"]
for char_to_remove in removal_list:
    data_cleaned = data_cleaned.replace(char_to_remove, " ")

data_cleaned = data_cleaned.lower()

print(len(data_cleaned))

characters = sorted(list(set(data_cleaned)))
id_to_character = {i: char for i, char in enumerate(characters)}
character_to_id = {char: i for i, char in enumerate(characters)}


def data_to_sequence(data, data_to_id_dict):
    seq_Xs, seq_Ys = list(), list()

    for i in range(0, len(data) - SEQ_LENGTH):
        seq = data[i:i + SEQ_LENGTH]
        label = data[i + SEQ_LENGTH]

        seq_Xs.append([data_to_id_dict[char] for char in seq])
        seq_Ys.append(data_to_id_dict[label])

    return seq_Xs, seq_Ys


seq_Xs, seq_ys = data_to_sequence(data_cleaned, character_to_id)

for x, y in zip(seq_Xs[0:2], seq_ys[0:2]):
    print(x, y)

train_X = np.reshape(seq_Xs, (len(seq_Xs), SEQ_LENGTH, FEATURE_NUM))
train_y = keras.utils.to_categorical(seq_ys)

train_X = train_X / float(len(characters))

model = Sequential([
    LSTM(700, input_shape=(
        train_X.shape[1], train_X.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(700),
    Dropout(0.2),
    Dense(train_y.shape[1], activation='softmax')
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_X, train_y,
                    epochs=50,
                    batch_size=128,
                    verbose=1,
                    shuffle=False)

full_string = [id_to_character[value] for value in string_mapped]

for i in range(1000):
    x = np.reshape(string_mapped, (1, len(string_mapped), 1))
    x = x / float(len(characters))

    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [id_to_character[value] for value in string_mapped]
    full_string.append(id_to_character[pred_index])

    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]

generated_text = ""
for char in full_string:
    generated_text += char

print(generated_text)
