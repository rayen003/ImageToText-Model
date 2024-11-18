import os
import random
from PIL import Image

# Directories
images_dir = "Images"
small_images_dir = "Small_Images"
captions_file = "captions.txt"
small_captions_file = "small_captions.txt"

# Create the directory for small images if it doesn't exist
os.makedirs(small_images_dir, exist_ok=True)

# List all images in the directory
images = os.listdir(images_dir)

# Calculate 1% of the total images
images_to_keep = max(1, len(images) // 100)

# Select 1% of the images randomly
sampled_images = random.sample(images, images_to_keep)

# Copy the sampled images to the new directory
for image in sampled_images:
    img = Image.open(f'{images_dir}/{image}')
    img.save(f'{small_images_dir}/{image}')

# Read the original captions file
with open(captions_file, 'r') as file:
    lines = file.readlines()

# Filter captions for the sampled images
header = lines[0]  # Keep the header
sampled_captions = [header] + [line for line in lines[1:] if line.split(',')[0] in sampled_images]

# Write the sampled captions to a new file
with open(small_captions_file, 'w') as file:
    file.writelines(sampled_captions)

print(f'Sampled {images_to_keep} images out of {len(images)} total images.')
print(f'Sampled captions written to {small_captions_file}.')