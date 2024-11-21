import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dropout, LSTM, Dense, add

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim=128, lstm_units=256, dense_units=256, dropout_rate=0.4):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.dropout1 = Dropout(dropout_rate)
        self.lstm = LSTM(lstm_units)
        self.dense1 = Dense(dense_units, activation='relu')
        self.dense2 = Dense(vocab_size, activation='softmax')

    def call(self, inputs1, inputs2):
        x1 = self.embedding(inputs2)
        x1 = self.dropout1(x1)
        x1 = self.lstm(x1)
        x = add([inputs1, x1])
        x = self.dense1(x)
        x = self.dense2(x)
        return x

# Example usage
if __name__ == "__main__":
    vocab_size = 5000  # Example vocabulary size
    feature_shape = 256  # Example feature shape from encoder
    max_length = 49  # Example maximum length of captions
    decoder = Decoder(vocab_size)
    # Build the model by calling it with sample inputs
    sample_feature_input = tf.random.uniform((1, feature_shape))
    sample_caption_input = tf.random.uniform((1, max_length), maxval=vocab_size, dtype=tf.int32)
    decoder(sample_feature_input, sample_caption_input)
    decoder.summary()