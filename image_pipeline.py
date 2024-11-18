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


import warnings


warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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
            features = tf.reshape(features, (-1)) # Flatten the features

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
        Returns image names and features as numpy arrays.
        """
        image_paths = self.get_image_paths()  # Step 1: Get image paths
        self.extract_image_features(image_paths)  # Step 2: Extract image features
        
        # Convert the dictionary to numpy arrays
        image_names = np.array(list(self.image_features.keys()))
        image_features = np.array(list(self.image_features.values()))
        
        # print(f"Image names shape: {image_names.shape}")
        # print(f"Image features shape: {image_features.shape}")
        
        return image_names, image_features

# # Usage
# Usage
#image_dir = 'Small_Images'
#pipeline = ImagePreprocessingPipeline(image_dir)

image_names, image_features = pipeline()

# Selecting a random image and its features
random_index = random.randint(0, len(image_names) - 1)
random_image_name = image_names[random_index]
random_image_features = image_features[random_index]

# Print the random image
image_path = os.path.join(image_dir, random_image_name)
image = Image.open(image_path + '.jpg')
image.show()

print(f'Randomly selected image: {random_image_name}')
print(f'First 5 features of the selected image: {random_image_features[:5]}')

