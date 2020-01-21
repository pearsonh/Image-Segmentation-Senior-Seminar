from PIL import Image
import numpy as np
import statistics as stat
import math

def regionSplit(image, threshold):
    imageGrey = image.convert(mode='L')
    imgArray = np.array(imageGrey).astype('int')
    splitGraph = np.array(split(imgArray, threshold))
    return merge(splitGraph, threshold)

def split(img, threshold):
    if flatTest(img, threshold):
        return img
    else:
        splitNumpy = []
        splitNumpy.append(split(img[:len(img)//2, :len(img)//2], threshold))
        splitNumpy.append(split(img[:len(img)//2, len(img)//2:], threshold))
        splitNumpy.append(split(img[len(img)//2:, :len(img)//2], threshold))
        splitNumpy.append(split(img[len(img)//2:, len(img)//2:], threshold))
        return splitNumpy

def merge(splitGraph, threshold):
    return splitGraph

def flatTest(array, threshold):
    x, y = len(array), len(array[0])
    mean = []
    for j in range(x):
        for k in range(y):
            mean.append(array[j,k])
    if len(mean) <= 4 or math.sqrt(stat.variance(mean, None)) <= threshold:
        return True
    else:
        return False


if __name__ == '__main__':
    img = Image.open("22093.jpg")
    print(regionSplit(img, 50)[0])
