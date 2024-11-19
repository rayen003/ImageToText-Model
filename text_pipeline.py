import os
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
import random
import matplotlib.pyplot as plt
from PIL import Image
import os


class TextPreprocessingPipeline:
    def __init__(self, text_dir, max_length):
        self.text_dir = text_dir
        self.tokenizer = Tokenizer()
        self.max_length = max_length

    def open_captions(self):
        with open(self.text_dir, 'r') as f:
            next(f)
            captions_doc = f.read()
        return captions_doc

    def create_dict(self, captions_doc):
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
        return mapping

    def clean(self, mapping):
        for key, captions in mapping.items():
            for i in range(len(captions)):
                # take one caption at a time
                caption = captions[i]
                # preprocessing steps
                # convert to lowercase
                caption = caption.lower()
                # delete digits, special chars, etc.
                caption = caption.replace('[^A-Za-z]', '')
                # delete additional spaces
                caption = caption.replace('\s+', ' ')
                # add start and end tags to the caption
                caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
                captions[i] = caption
        return mapping

    def get_data(self, mapping):
        all_captions = []
        for key in mapping.keys():
            [all_captions.append(caption) for caption in mapping[key]]
        return all_captions

    def __call__(self):
        captions_doc = self.open_captions()
        mapping = self.create_dict(captions_doc)
        data = self.clean(mapping)
        all_captions = self.get_data(data)
        self.tokenizer.fit_on_texts(all_captions)
        vocab_size = len(self.tokenizer.word_index) + 1
        return self.tokenizer, vocab_size, data, all_captions



