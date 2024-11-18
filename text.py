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


"""# Images preprocessing

"""

import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from PIL import Image
import random
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer

import warnings

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# ---- IMAGE PREPROCESSING PIPELINE ----
class ImagePreprocessingPipeline:
    def __init__(self, image_dir, img_height=224, img_width=224):
        self.image_dir = image_dir
        self.img_height = img_height
        self.img_width = img_width
        self.image_features = {}

        # Define a simple CNN for feature extraction
        self.cnn_model = self.build_cnn_model()

    def build_cnn_model(self):
        """
        Build a simple CNN model for feature extraction.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.GlobalAveragePooling2D()
        ])
        return model

    def load_image(self, image_path):
        """
        Load and preprocess an image.
        """
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (self.img_height, self.img_width))
        img = img / 255.0  # Normalize to [0,1]
        return img

    def extract_image_features(self, image_paths):
        """
        Process each image path, extract features using the CNN, and store them in a dictionary.
        """
        for image_path in tqdm(image_paths):
            img = self.load_image(image_path)
            img = tf.expand_dims(img, axis=0)  # Add batch dimension

            # Extract features using the CNN model
            features = self.cnn_model(img)
            features = tf.reshape(features, (-1))  # Flatten the features

            image_name = os.path.splitext(os.path.basename(image_path))[0]
            self.image_features[image_name] = features.numpy()

    def get_image_paths(self):
        """
        Get all image paths from the specified directory.
        """
        return [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir) if file.endswith('.jpg')]

    def __call__(self):
        """
        Streamline the entire process of image preprocessing by calling the methods in sequence.
        Returns image names and features as a dictionary.
        """
        image_paths = self.get_image_paths()  # Step 1: Get image paths
        self.extract_image_features(image_paths)  # Step 2: Extract image features

        return self.image_features

# Usage
image_dir = 'Small_Images'
pipeline = ImagePreprocessingPipeline(image_dir)
image_features_dict = pipeline()
BASE_DIR = "small_captions.txt"

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
max_length
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
epochs = 20
steps = len(data) // batch_size
model.fit(dataset, epochs=epochs, steps_per_epoch=steps, verbose=1)



def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
    return in_text

from nltk.translate.bleu_score import corpus_bleu
# validate with test data
actual, predicted = list(), list()

for key in tqdm(data.keys()):
    # get actual caption
    captions = data[key]
    # predict the caption for image
    y_pred = predict_caption(model, image_features_dict[key], tokenizer, max_length)
    # split into words
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    # append to the list
    actual.append(actual_captions)
    predicted.append(y_pred)
# calcuate BLEU score
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))