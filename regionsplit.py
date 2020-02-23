from PIL import Image
import numpy as np
import statistics as stat
import math
import sys
from eval import region_based_eval

def regionSplitAndMerge(image, threshold):
    imageGrey = image.convert(mode='L')
    img = np.array(imageGrey).astype("int")
    blank = np.empty(img.shape)
    splitMergeSeg = splitAndMerge(img, blank, threshold)
    #splitSeg = split(img, np.empty(img.shape), threshold)
    return splitMergeSeg

def splitAndMerge(img, blank, threshold):
    tL = img[:img.shape[0]//2, :img.shape[1]//2]
    tR = img[:img.shape[0]//2, img.shape[1]//2:]
    bL = img[img.shape[0]//2:, :img.shape[1]//2]
    bR = img[img.shape[0]//2:, img.shape[1]//2:]
    if flatTest(tL, threshold) and flatTest(tR, threshold) and flatTest(bL, threshold) and flatTest(bR, threshold):
        #if flatTest(img, threshold):
        x,y = img.shape
        avgVal = np.mean(img)
        for i in range(x):
            for j in range(y):
                blank[i][j] = avgVal
    else:
        bHeight = blank.shape[1]
        bWidth = blank.shape[0]
        splitAndMerge(tL, blank[:bWidth//2, :bHeight//2], threshold)
        splitAndMerge(tR, blank[:bWidth//2, bHeight//2:], threshold)
        splitAndMerge(bL, blank[bWidth//2:, :bHeight//2], threshold)
        splitAndMerge(bR, blank[bWidth//2:, bHeight//2:], threshold)
        if flatTest(np.concatenate([tL, tR], axis=1), threshold):
            if flatTest(np.concatenate([tL, bL], axis=0), threshold):
                blank[:bWidth//2, :bHeight//2] = np.full(tL.shape, np.mean(np.concatenate([tL, tR], axis=1)))
                blank[:bWidth//2, bHeight//2:] = np.full(tR.shape, np.mean(np.concatenate([tL, tR], axis=1)))
                blank[bWidth//2:, :bHeight//2] = np.full(bL.shape, np.mean(np.concatenate([tL, tR], axis=1)))
            else:
                blank[:bWidth//2, :bHeight//2] = np.full(tL.shape, np.mean(np.concatenate([tL, tR], axis=1)))
                blank[:bWidth//2, bHeight//2:] = np.full(tR.shape, np.mean(np.concatenate([tL, tR], axis=1)))
        if flatTest(np.concatenate([bL, bR], axis=1), threshold):
            if flatTest(np.concatenate([tR, bR], axis=0), threshold):
                blank[bWidth//2:, :bHeight//2] = np.full(bL.shape, np.mean(np.concatenate([bL, bR], axis=1)))
                blank[bWidth//2:, bHeight//2:] = np.full(bR.shape, np.mean(np.concatenate([bL, bR], axis=1)))
                blank[:bWidth//2, bHeight//2:] = np.full(tR.shape, np.mean(np.concatenate([bL, bR], axis=1)))
            else:
                blank[bWidth//2:, :bHeight//2] = np.full(bL.shape, np.mean(np.concatenate([bL, bR], axis=1)))
                blank[bWidth//2:, bHeight//2:] = np.full(bR.shape, np.mean(np.concatenate([bL, bR], axis=1)))
        elif flatTest(np.concatenate([tR, bR], axis=0), threshold):
            blank[:bWidth//2, bHeight//2:] = np.full(tR.shape, np.mean(np.concatenate([tR, bR], axis=0)))
            blank[bWidth//2:, bHeight//2:] = np.full(bR.shape, np.mean(np.concatenate([tR, bR], axis=0)))
        elif flatTest(np.concatenate([tL, bL], axis=0), threshold):
            blank[:bWidth//2, :bHeight//2] = np.full(tL.shape, np.mean(np.concatenate([tL, bL], axis=0)))
            blank[bWidth//2:, :bHeight//2] = np.full(bL.shape, np.mean(np.concatenate([tL, bL], axis=0)))
    return blank


def flatTest(array, threshold):
    x, y = array.shape
    mean = []
    threshold = threshold #/ ((x * y) / 2)
    for j in range(x):
        for k in range(y):
            mean.append(array[j,k])
    if len(mean) <= 4 or stat.variance(mean, None) <= threshold:
        return True
    else:
        return False

def eraseEdge(edges, bounds, axis=0):
    x,y = edges.shape
    i, j = bounds[0]
    if axis == 1:
        while (i, j) != bounds[1]:
            print(edges[i][j])
            edges[i][j] = 0
            if i-1 < x:
                edges[i-1][j] = 0
            j += 1
    if axis == 0:
        while (i, j) != bounds[1]:
            print(edges[i][j])
            edges[i][j] = 0
            if j-1 < y:
                edges[i][j-1] = 0
            i += 1
    return edges

if __name__ == '__main__':
    img = Image.open("22093.jpg")
    array = regionSplitAndMerge(img, 450)
    splitAndMerge = array
    newImage = Image.fromarray(splitAndMerge)
    newImage.show()
