from PIL import Image
import numpy as np
import statistics as stat
import math
from scipy import spatial as sc

class regionSplitTree:
    def __init__(self, root):
        self.root = root
        self.size = 0

    def split(self, threshold):
        return self.root.split(threshold)

    def merge(self, threshold):
        return self.root.merge(threshold)

class quadNode:
    def __init__(self, data, parent=None, tr=None, tl=None, br=None, bl=None):
        self.data = data
        self.topRight = tr
        self.topLeft = tl
        self.bottomRight = br
        self.bottomLeft = bl

    def flatTest(self, array, threshold):
        x, y = len(array), len(array[0])
        mean = []
        for j in range(x):
            for k in range(y):
                mean.append(array[j,k])
        if len(mean) <= 4 or math.sqrt(stat.variance(mean, None)) <= threshold:
            return True
        else:
            return False

    def split(self, threshold):
        if self.flatTest(self.data, threshold):
            return self
        else:
            self.topRight = quadNode(self.data[:len(self.data)//2, :len(self.data)//2], self)
            self.topRight.split(threshold)
            self.topLeft = quadNode(self.data[:len(self.data)//2, len(self.data)//2:], self)
            self.topLeft.split(threshold)
            self.bottomRight = quadNode(self.data[len(self.data)//2:, :len(self.data)//2], self)
            self.bottomRight.split(threshold)
            self.bottomLeft = quadNode(self.data[len(self.data)//2:, len(self.data)//2:], self)
            self.bottomLeft.split(threshold)
            self.data = []
            return self


def regionSplit(image, threshold):
    imageGrey = image.convert(mode='L')
    '''imgTree = regionSplitTree(quadNode(np.array(imageGrey).astype('int')))
    print(imgTree.root)
    imgTree.split(threshold)
    imgTree.merge(threshold)'''
    img = np.array(imageGrey).astype("int")
    imgTree = split(img, np.empty(img.shape), threshold)
    return imgTree

def split(img, blank, threshold):
    if flatTest(img, threshold):
        return blank
    else:
        x,y = img.shape
        '''print(str(x), str(y))'''
        regVal = img[0][0]
        for i in range(x):
            for j in range(y):
                if i <= x//2 and j < y//2:
                    blank[i][j] = np.mean(img[:len(img)//2, :len(img)//2])
                elif i <= x//2 and j > y//2:
                    blank[i][j] = np.mean(img[:len(img)//2, len(img)//2:])
                elif i > x//2 and j <= y//2:
                    blank[i][j] = np.mean(img[len(img)//2:, :len(img)//2])
                elif i > x//2 and j >= y//2:
                    blank[i][j] = np.mean(img[len(img)//2, len(img)//2:])
        split(img[:len(img)//2, :len(img)//2], blank[:len(blank)//2, :len(blank)//2], threshold)
        split(img[:len(img)//2, len(img)//2:], blank[:len(blank)//2, len(blank)//2:], threshold)
        split(img[len(img)//2:, :len(img)//2], blank[len(blank)//2:, :len(blank)//2], threshold)
        split(img[len(img)//2:, len(img)//2:], blank[len(blank)//2:, len(blank)//2:], threshold)
        return blank

def merge(splitGraph, threshold, imageArray):
    x, y = imageArray.shape
    mergeImage = np.ones((x, y))*100
    for i in range(len(splitGraph)-1):
        layer = splitGraph
        while type(layer) != np.ndarray:
            layer = layer[i]
    return mergeImage

def flatTest(array, threshold):
    x, y = len(array), len(array[0])
    mean = []
    for j in range(x):
        for k in range(y):
            mean.append(array[j,k])
    if len(mean) <= 4 or stat.variance(mean, None) <= threshold:
        return True
    else:
        '''print(math.sqrt(stat.variance(mean, None)))'''
        return False


if __name__ == '__main__':
    img = Image.open("22093.jpg")
    newImage = Image.fromarray(regionSplit(img, 1000))
    newImage.show()
