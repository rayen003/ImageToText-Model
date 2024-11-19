import os
from tqdm import tqdm
from PIL import Image
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Embedding, add
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import warnings
warnings.filterwarnings("ignore")



# Usage
image_dir = 'Images'
pipeline = ImagePreprocessingPipeline(image_dir)
image_features_dict = pipeline()
BASE_DIR = "captions.txt"

with open(BASE_DIR, 'r') as f:
    next(f)
    captions_doc = f.read()

# create a dictionary to map image id with captions
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)


def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc., 
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption
    return mapping



data = clean(mapping)


# create a list of all captions
all_captions = []
for key in data.keys():
    [all_captions.append(caption) for caption in data[key]]


tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1




max_length = max(len(caption.split()) for caption in all_captions)
inputs1 = Input(shape=(None,128))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# decoder (feed forward) layers
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# summarize model
#print(model.summary())

# define parameters for data generator
data_keys= data.keys()
mapping=data
features= image_features_dict
batch_size=32
max_length=50

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, LSTM, add
from tensorflow.keras.models import Model

# Assuming data, tokenizer, and image_features_dict are already defined

# Create a list of all captions
all_captions = []
for key in data.keys():
    [all_captions.append(caption) for caption in data[key]]

# Tokenize the captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

# Determine the maximum length of the captions
max_length = max(len(caption.split()) for caption in all_captions)

# Define the data generator function
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while True:
        for key in data_keys:
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(features[key])
                    X2.append(in_seq)
                    y.append(out_seq)
                    n += 1
                    if n == batch_size:
                        X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                        yield (X1, X2), y
                        X1, X2, y = list(), list(), list()
                        n = 0

# Define parameters for data generator
data_keys = list(data.keys())
mapping = data
features = image_features_dict
batch_size = 32

# Define the output signature for the dataset
output_signature = (
    (tf.TensorSpec(shape=(None, features[list(features.keys())[0]].shape[0]), dtype=tf.float32),
     tf.TensorSpec(shape=(None, max_length), dtype=tf.float32)),
    tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)
)

# Create the dataset using from_generator
dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size),
    output_signature=output_signature
)

# Define the model
inputs1 = Input(shape=(features[list(features.keys())[0]].shape[0],))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Fit the model
epochs = 1
steps = len(data) // batch_size
model.fit(dataset, epochs=epochs, steps_per_epoch=steps, verbose=1)




def predict(sequence, feature, tokenizer, model, max_length):
    in_text = sequence
    for _ in range(max_length):
        # prepare the sequence
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        # prepare the image
        image = np.array([feature])
        # predict the next word
        yhat = model.predict([image, seq], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = tokenizer.index_word.get(yhat)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text

# Randomly select an image feature
random_key = random.choice(list(image_features_dict.keys()))
image_feature = image_features_dict[random_key]

# Generate the caption
pred = predict('startseq', image_feature, tokenizer, model, max_length)
print(f"Generated caption: {pred}")