import tensorflow as tf
import warnings
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

class Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu', name='dense_1_relu')
        self.dense2 = tf.keras.layers.Dense(embedding_dim, name='dense_2_embedding')
        
    def __call__(self, x):
        x = self.dense1(x)
        return self.dense2(x)

    def inspect_encoder(self, input_shape):
        """
        Inspects the encoder by building the model with the given input shape.
        """
        self.build(input_shape)
        return self.summary()


