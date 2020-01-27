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

        return

class quadNode:
    def __init__(self, data, parent=None, tr=None, tl=None, br=None, bl=None):
        self.data = data
        self.topRight = tr
        self.topLeft = tl
        self.bottomRight = br
        self.bottomLeft = bl

    def split(self, threshold):
        if flatTest(self.data, threshold):
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


def regionSplit(image, threshold):
    imageGrey = image.convert(mode='L')
    imgTree = regionSplitTree(quadNode(np.array(imageGrey).astype('int')))
    return imgTree.split(threshold)

'''def split(img, threshold):
    if flatTest(img, threshold):
        return img
    else:
        splitGraph = []
        splitGraph.append(split(img[:len(img)//2, :len(img)//2], threshold))
        splitGraph.append(split(img[:len(img)//2, len(img)//2:], threshold))
        splitGraph.append(split(img[len(img)//2:, :len(img)//2], threshold))
        splitGraph.append(split(img[len(img)//2:, len(img)//2:], threshold))
        return splitGraph'''

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
    if len(mean) <= 4 or math.sqrt(stat.variance(mean, None)) <= threshold:
        return True
    else:
        return False


if __name__ == '__main__':
    img = Image.open("22093.jpg")
    print(regionSplit(img, 50).data)
