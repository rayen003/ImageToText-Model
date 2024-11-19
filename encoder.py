import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Embedding, add, Input, Dropout
from tensorflow.keras.models import Model

class Encoder(tf.keras.Model):
    def __init__(self, dropout_rate=0.4, dense_units=256):
        super(Encoder, self).__init__()
        self.fe1 = Dropout(dropout_rate)
        self.fe2 = Dense(dense_units, activation='relu')

    def call(self, inputs):
        fe1 = self.fe1(inputs)
        fe2 = self.fe2(fe1)
        return fe2
