# main.py
import numpy as np
from image_pipeline import ImagePreprocessingPipeline
from text_pipeline import TextPreprocessingPipeline
from encoder import Encoder
from decoder import SimpleDecoder
import tensorflow as tf

def prepare_data_for_model(image_dir, text_path, img_height=224, img_width=224, max_len=50, embed_dim=128):
    # Initialize pipelines
    image_pipeline = ImagePreprocessingPipeline(image_dir, img_height=img_height, img_width=img_width)
    text_pipeline = TextPreprocessingPipeline(max_len=max_len, embed_dim=embed_dim)
    
    # Step 1: Extract image features and captions
    image_names, image_features = image_pipeline()
    embeddings_np, text_images, cleaned_data = text_pipeline(text_path)
    
    # Step 2: Repeat each image feature 5 times to match the number of captions
    X_image = np.repeat(image_features, repeats=5, axis=0)
    
    # Step 3: Convert caption embeddings to input and output sequences
    X_seq = embeddings_np[:, :-1, :]  # Input sequence shifted to the left by 1 token
    y_seq = embeddings_np[:, 1:, :]   # Target sequence shifted to the right by 1 token

    # Verification of shapes
    print(f"Image names shape: {image_names.shape}")          # (80,)
    print(f"Image features shape: {image_features.shape}")    # (80, 128)
    print(f"Embeddings shape: {embeddings_np.shape}")         # (400, 50, 128)
    print(f"Repeated image features shape (X_image): {X_image.shape}")  # (400, 128)
    print(f"Input sequence shape (X_seq): {X_seq.shape}")     # (400, 49, 128)
    print(f"Target sequence shape (y_seq): {y_seq.shape}")    # (400, 49, 128)
    
    return X_image, X_seq, y_seq

class CaptioningModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units):
        super(CaptioningModel, self).__init__()
        self.encoder = Encoder(embedding_dim)
        self.decoder = SimpleDecoder(vocab_size, embedding_dim, lstm_units)
        
    def call(self, inputs):
        """
        inputs: Tuple of (image_features, input_sequence)
            - image_features: Encoded features from the image pipeline, shape (batch_size, 128)
            - input_sequence: Token embeddings from text pipeline, shape (batch_size, 49, 128)
        """
        image_features, input_sequence = inputs
        
        # Encode the image features
        encoded_features = self.encoder(image_features)  # Shape (batch_size, embedding_dim)
        
        # Initialize the hidden state for the decoder
        decoder_hidden, decoder_cell = None, None
        predictions = []
        
        # Loop over each timestep in the sequence
        for t in range(input_sequence.shape[1]):
            # Get the embedding for the current timestep
            input_token = input_sequence[:, t, :]
            
            # Forward pass through the decoder
            pred, decoder_hidden, decoder_cell = self.decoder(
                input_token, input_embeddings=embedding_dim ,initial_state=(decoder_hidden, decoder_cell)
            )
            
            predictions.append(pred)
        
        # Stack predictions along the time dimension
        return tf.stack(predictions, axis=1)  # Shape (batch_size, seq_len, vocab_size)

# Model hyperparameters
vocab_size = 500
embedding_dim = 128
lstm_units = 256
BATCH_SIZE = 32
EPOCHS = 10

# Instantiate the captioning model
model = CaptioningModel(vocab_size, embedding_dim, lstm_units)
print("Captioning model defined successfully.")

# Usage
image_dir = 'C:\\Captions_generator\\Small_Images'
text_path = 'C:\\Captions_generator\\small_captions.txt'

# Prepare data using the function defined
X_image, X_seq, y_seq = prepare_data_for_model(image_dir, text_path)

# Compile and train the model
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(image_features, input_sequence, target_sequence):
    with tf.GradientTape() as tape:
        predictions = model((image_features, input_sequence))
        loss = loss_object(target_sequence, predictions)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# Training loop
dataset = tf.data.Dataset.from_tensor_slices((X_image, X_seq, y_seq)).batch(BATCH_SIZE)

for epoch in range(EPOCHS):
    total_loss = 0
    for batch, (batch_X_image, batch_X_seq, batch_y_seq) in enumerate(dataset):
        batch_loss = train_step(batch_X_image, batch_X_seq, batch_y_seq)
        total_loss += batch_loss
    
    print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / batch:.4f}')

    C:\Captions_generator\image_pipeline.py