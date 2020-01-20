from PIL import Image
import numpy as np
import statistics as stat

def regionSplit(image, threshold):
    imageGrey = image.convert(mode='L')
    imgArray = np.array(imageGrey).astype('int')
    splitArray = np.array(split(imgArray, threshold))
    print(splitArray.shape)
    print(splitArray[0])
    return merge(splitArray, threshold)

def split(img, threshold):
    if flatTest(img, threshold):
        return img
    else:
        return (split(img[:len(img)//2, 0:len(img)//2], threshold), split(img[:len(img)//2, 0:len(img)//2], threshold),  split(img[len(img)//2:, 0:len(img)//2], threshold), split(img[len(img)//2:, len(img)//2:], threshold))

def merge(img, threshold):

    return

def flatTest(array, threshold):
    x, y = len(array), len(array[0])
    mean = []
    for j in range(x):
        for k in range(y):
            mean.append(array[j,k])
    if len(mean) <= 4 or stat.variance(mean, None) <= threshold:
        return True
    else:
        return False


if __name__ == '__main__':
    img = Image.open("22093.jpg")
    new_img = regionSplit(img, 5000)
