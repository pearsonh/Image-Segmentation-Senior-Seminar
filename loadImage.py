import numpy as np
from PIL import Image
import scipy

def load_image(filename):
    img = Image.open(filename)
    img.load()
    returnimg = np.asarray(img,dtype="int32")
    img.close()
    return returnimg

def save_image(filename,image):
    scipy.misc.imsave(filename,image)
