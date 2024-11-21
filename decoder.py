import tensorflow as tf

class SimpleDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units):
        # LSTM layer for sequential processing
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        # Dense layer to map LSTM outputs to vocabulary for prediction
        self.dense = tf.keras.layers.Dense(vocab_size)
        # Project image features to match embedding dimensions
        self.image_proj = tf.keras.layers.Dense(embedding_dim, activation='relu')
        
    def call(self, image_features, input_embeddings, initial_state=None):
        """
        image_features: Encoded image features (e.g., output of Encoder)
        input_embeddings: Sequence of embeddings (caption tokens with teacher forcing)
        initial_state: Optional initial LSTM state
        """
        # Project image features to match input embedding dimension
        projected_image_features = self.image_proj(image_features)
        # Expand dims to concatenate along the sequence axis
        projected_image_features = tf.expand_dims(projected_image_features, 1)
        
        # Concatenate image features to the start of the input sequence
        decoder_inputs = tf.concat([projected_image_features, input_embeddings], axis=1)
        
        # Pass through LSTM layer
        lstm_output, state_h, state_c = self.lstm(decoder_inputs, initial_state=initial_state)
        
        # Predict next word from LSTM output
        output = self.dense(lstm_output)
        
        return output, state_h, state_c