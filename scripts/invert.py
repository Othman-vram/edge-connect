import os
from PIL import Image, ImageOps

# Supported image extensions
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

def invert_image(path):
    try:
        img = Image.open(path).convert("L")  # Convert to grayscale
        inverted = ImageOps.invert(img)
        inverted.save(path)  # Overwrite the original image
        print(f"Inverted: {path}")
    except Exception as e:
        print(f"Failed to process {path}: {e}")

def walk_and_invert(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                full_path = os.path.join(root, file)
                invert_image(full_path)

# Replace this with your target directory
target_directory = ""

walk_and_invert(target_directory)
