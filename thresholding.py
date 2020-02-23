from PIL import Image
import numpy as np

def baseline_thresholding(image):
    #Creates a greyscale version of the input image
    imageGrey = image.convert(mode="L")
    x, y = imageGrey.size
    #Gets a threshold for the image
    threshold = get_greyscale_threshold(imageGrey)
    returnImage = np.zeros(shape=(y,x))
    #Maps 1's and 0's based on whether the pixel values are above or below the threshold.
    for j in range(x):
        for i in range(y):
            returnImage[i,j] = 1 if imageGrey.getpixel((j,i)) > threshold else 0
    return returnImage

#Determines a threshold via an average of all pixels multipled by 1.5
def get_greyscale_threshold(image):
    x,y = image.size
    averageGrey = 0
    for i in range(x):
        for j in range(y):
            averageGrey += image.getpixel((i,j))
    averageGrey /= (x * y)
    return averageGrey*1.5

if __name__ == "__main__":
    img = Image.open("22093.jpg")
    img = Image.fromarray(baseline_thresholding(img)*200)
    img.show()
