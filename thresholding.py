from PIL import Image
import numpy as np

def baseline_thresholding(img):
    img_grey = img.convert(mode="L")
    x, y = img_grey.size
    threshold = get_greyscale_threshold(img_grey)
    returnimg = np.zeros(shape=(y,x))
    for j in range(x):
        for i in range(y):
            returnimg[i,j] = 1 if img_grey.getpixel((j,i)) > threshold else 0
    return returnimg

def get_greyscale_threshold(img):
    x,y = img.size
    average_grey = 0
    for i in range(x):
        for j in range(y):
            average_grey += img.getpixel((i,j))
    average_grey /= (x * y)
    return average_grey*1.5

if __name__ == "__main__":
    img = Image.open("22093.jpg")
    img = Image.fromarray(baseline_thresholding(img)*200)
    img.show()
