import tensorflow as tf

class SimpleDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units):
        super(SimpleDecoder, self).__init__()
        # LSTM layer for sequential processing
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        # Dense layer to map LSTM outputs to vocabulary for prediction
        self.dense = tf.keras.layers.Dense(vocab_size)
        # Project image features to match embedding dimensions
        self.image_proj = tf.keras.layers.Dense(embedding_dim, activation='relu')
        
    def __call__(self, image_features, input_embeddings, initial_state=None):
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
        predictions = self.dense(lstm_output)
        
        return predictions, state_h, state_c

# Test values for embedding dimension and vocab size
embedding_dim = 128
vocab_size = 500
lstm_units = 256

# Create decoder instance
decoder = SimpleDecoder(vocab_size, embedding_dim, lstm_units)

# Example inputs
image_features_sample = tf.random.normal([400, embedding_dim])  # Example encoded image features
input_embeddings_sample = tf.random.normal([400, 49, embedding_dim])  # Example input embeddings (teacher forcing)

# Forward pass
predictions, state_h, state_c = decoder(image_features_sample, input_embeddings_sample)

# Verify output shapes
print(f"Predictions shape: {predictions.shape}")
print(f"State_h shape: {state_h.shape}")
print(f"State_c shape: {state_c.shape}")
