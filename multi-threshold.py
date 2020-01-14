from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu

def otsu_greyspace_thresholding(image):
    #Creates a greyscale version of the input image
    imageGrey = image.convert(mode="L")

    x, y = imageGrey.size
    #Gets a threshold for the image
    threshold = get_histogram_threshold(imageGrey)[0]
    print(threshold)
    returnImage = np.zeros(shape=(y,x))
    #Maps 1's and 0's based on whether the pixel values are above or below the threshold.
    for j in range(x):
        for i in range(y):
            returnImage[i,j] = 1 if imageGrey.getpixel((j,i)) > threshold else 0
    return returnImage

def otsu_color_thresholding(image):

    red = get_color_thresholds(image, "R")
    blue = get_color_thresholds(image, "G")
    green = get_color_thresholds(image, "B")
    return (create_image(red, image, "R"), create_image(blue, image, "G"),create_image(green, image, "B"))


def get_histogram_threshold(image):
    a = np.array(image.histogram())
    print(a)
    return np.where(a == threshold_otsu(a))

def create_image(threshold, image, color):
    threshold_color = 0
    if color == "R":
        threshold_color = 0
    elif color == "G":
        threshold_color = 1
    else:
        threshold_color = 2
    x, y = image.size
    returnImage = np.zeros(shape=(y,x))
    for j in range(x):
        for i in range(y):
            returnImage[i,j] = 1 if image.getpixel((j,i)) > threshold else 0
    return returnImage


def get_color_thresholds(image, color):
    a = []
    if color == "R":
        a = np.array(image.histogram()[0:256])
    elif color == "G":
        a = np.array(image.histogram()[256:512])
    else:
        a = np.array(image.histogram()[512:768])
    return np.where(a == threshold_otsu(a))

if __name__ == "__main__":
    img = Image.open("test_image.jpg")
    new_image = Image.fromarray(otsu_greyspace_thresholding(img)*200)
    new_image.show()
    images = []
    color_thresholding_images = otsu_color_thresholding(img)
    for segments in color_thresholding_images:
        new_image = Image.fromarray(segments * 200)
        new_image.show()
        images.append(new_image)
    new_image = Image.merge("RGB", tuple(images))
    new_image.show();
