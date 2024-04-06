from PIL import Image
import numpy as np

def save_img(rgb_array, filename):
    img = Image.fromarray(rgb_array.astype(np.uint8))
    img.save(filename)
