"""Image file discovery, safe loading, and perceptual hashing."""
import os

import imagehash
from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def list_image_files(directory, extensions=IMAGE_EXTENSIONS):
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in extensions:
                files.append(os.path.join(dirpath, filename))
    return files


# open an image, returning None instead of raising on corrupted/unreadable files
def safe_open_image(image_path):
    try:
        return Image.open(image_path)
    except Exception:
        return None


def compute_phash(image_path):
    image = safe_open_image(image_path)
    if image is None:
        return None
    return str(imagehash.phash(image))
