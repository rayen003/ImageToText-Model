import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, Input

class Encoder(tf.keras.Model):
    def __init__(self, feature_shape, dense_units=256, dropout_rate=0.4):
        super(Encoder, self).__init__()
        self.dropout = Dropout(dropout_rate)
        self.dense = Dense(dense_units, activation='relu')
        self.feature_shape = feature_shape

    def call(self, inputs):
        x = self.dropout(inputs)
        x = self.dense(x)
        return x

# Example usage
if __name__ == "__main__":
    feature_shape = 128  # Example feature shape
    encoder = Encoder(feature_shape)
    # Build the model by calling it with a sample input
    sample_input = tf.random.uniform((1, feature_shape))
    encoder(sample_input)
    encoder.summary()