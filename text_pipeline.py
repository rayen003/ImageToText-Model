import tensorflow as tf
import numpy as np
import re
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
import os
# disable warnings
warnings.filterwarnings("ignore")

os.environ['TF_ENABLE_ONEDNN_OPTS']= "0"

class TextPreprocessingPipeline:
    def __init__(self, max_len=50, embed_dim=128):
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.tokenizer = None
        self.vocab_size = 500
        self.embedding_layer = None

    def prepare_text(self, text_path):
        '''
        Read the text file, split into lines, and return images and captions.
        '''
        with open(text_path, 'r') as f:
            text = f.read()
        lines = text.split('\n')
        images, caps = [], []


        for line in lines:
            try:
                img, cap = line.split('.jpg,')
                images.append(img + '.jpg')
                caps.append(cap.strip())
            except:
                pass
        return images, caps

    def clean_text(self, text):
        '''
        Remove unwanted characters and convert to lowercase.
        '''
        return re.sub(r"[^a-zA-Z0-9]+", ' ', text.lower())

    def create_dictionary(self, images, captions):
        '''
        Create a dictionary of images and their corresponding captions.
        '''
        data = {}
        for img, cap in zip(images, captions):
            if img not in data:
                data[img] = [self.clean_text(cap)]
            else:
                data[img].append(self.clean_text(cap))
        return data

    def add_start_and_end_tokens(self, text):
        '''
        Add start and end tokens to a caption.
        '''
        return '<start> ' + text + ' <end>'

    def preprocess_captions(self, data):
        '''
        Add start and end tokens to all captions.
        '''
        for key in data:
            data[key] = [self.add_start_and_end_tokens(text) for text in data[key]]
        return data

    def tokenize_captions(self, captions):
        '''
        Tokenize and pad the captions.
        '''
        vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=self.vocab_size, output_mode='int', output_sequence_length=self.max_len,
                                                            pad_to_max_tokens=True)
        vectorize_layer.adapt(captions)
        self.tokenizer = vectorize_layer
        sequences = vectorize_layer(captions)


        return sequences

    def create_embedding_layer(self):
        '''
        Create an embedding layer.
        '''
        if self.vocab_size is None:
            raise ValueError("You must tokenize captions before creating the embedding layer.")
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim)
        return self.embedding_layer

    def get_embedding_output(self, padded_sequences):
        '''
        Generate embedding output for tokenized sequences.
        '''
        if self.embedding_layer is None:
            raise ValueError("You must create the embedding layer first.")

        return self.embedding_layer(padded_sequences)

    def __call__(self, text_path):
        '''
        Full pipeline that reads text, cleans it, tokenizes it, and returns the embeddings as a numpy array.
        Also returns the image names and cleaned captions.
        '''
        images, captions = self.prepare_text(text_path)
        data = self.create_dictionary(images, captions)
        cleaned_data = self.preprocess_captions(data)
        all_captions = [cap for sublist in cleaned_data.values() for cap in sublist]
        padded_sequences = self.tokenize_captions(all_captions)
        self.create_embedding_layer()
        embeddings = self.get_embedding_output(padded_sequences)
        
        # Convert embeddings to numpy array
        embeddings_np = embeddings.numpy()
        # print(f'embeddings shape: {embeddings_np.shape}')
        # print(f'vocab size: {len(self.tokenizer.get_vocabulary())}')

        return embeddings_np, images, cleaned_data


# # # Usage
# text_path = 'C:\Captions_generator\small_captions.txt'
# pipeline = TextPreprocessingPipeline(max_len=50, embed_dim=128)

# embeddings_np, images, cleaned_data = pipeline(text_path)

# random_key=random.choice(list(cleaned_data.keys()))
# print(f'key chosen is: {random_key}')

# print('---------------')
# print('cleaned data')
# print(cleaned_data[random_key])



