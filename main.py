from image_pipeline import ImagePreprocessingPipeline
from text_pipeline import TextPreprocessingPipeline
from encoder import Encoder
from decoder import Decoder
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')


image_dir = 'Images'
image_pipeline = ImagePreprocessingPipeline(image_dir)
image_features_dict = image_pipeline()

text_dir = 'captions.txt'
max_length = 49
text_pipeline = TextPreprocessingPipeline(text_dir, max_length)
tokenizer, vocab_size, data, all_captions = text_pipeline()

# Extract the shape of the image features
image_feature_shape = image_features_dict[list(image_features_dict.keys())[0]].shape[0]

# Define the Encoder
encoder = Encoder(image_feature_shape)

# Define the Decoder
embedding_dim = 128
lstm_units = 256
decoder = Decoder(vocab_size, embedding_dim, lstm_units)

# Define the inputs
image_input = Input(shape=(image_feature_shape,), name='image_input')
caption_input = Input(shape=(max_length,), name='caption_input')

# Get the encoded image features
encoded_image = encoder(image_input)

# Get the decoder outputs
decoder_outputs = decoder(encoded_image, caption_input)

# Define the model
model = Model(inputs=[image_input, caption_input], outputs=decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Print the model summary
model.summary()

# Define the data generator
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
                        yield (tf.convert_to_tensor(X1, dtype=tf.float32), tf.convert_to_tensor(X2, dtype=tf.float32)), tf.convert_to_tensor(y, dtype=tf.float32)
                        X1, X2, y = list(), list(), list()
                        n = 0

# Define parameters for data generator
data_keys = list(data.keys())
batch_size = 32

# Create the dataset using from_generator
dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(data_keys, data, image_features_dict, tokenizer, max_length, vocab_size, batch_size),
    output_signature=(
        (tf.TensorSpec(shape=(None, image_feature_shape), dtype=tf.float32),
         tf.TensorSpec(shape=(None, max_length), dtype=tf.float32)),
        tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)
    )
)

# Train the model
epochs = 50
steps_per_epoch = len(data_keys) // batch_size
model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)

# Save the model weights after training
model.save_weights('model_weights.weights.h5')

# Load the model weights before making predictions
# model.load_weights('model_weights.weights.h5')

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

# Test the model
sequence = 'startseq'
# random_key = random.choice(list(image_features_dict.keys()))
# print(f"key chosen is {random_key}")
feature = image_features_dict['191003287_2915c11d8e']
caption = predict(sequence, feature, tokenizer, model, max_length)
print(caption)


image_path = '/Users/rayengallas/Captions_Generator/Images/191003287_2915c11d8e.jpg'
image = Image.open(image_path)

# Convert the image to a NumPy array
image_array = np.array(image)

# Display the image
plt.imshow(image_array)
plt.axis('off')  # Hide the axis
plt.show()