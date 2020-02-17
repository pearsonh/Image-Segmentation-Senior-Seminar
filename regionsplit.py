from PIL import Image
import numpy as np
import statistics as stat
import math
import sys
from eval import region_based_eval

def regionSplitAndMerge(image, threshold):
    imageGrey = image.convert(mode='L')
    img = np.array(imageGrey).astype("int")
    #adjGraph = quadNode([img.shape, [0,0]])
    #print(adjGraph.regions)
    #print(adjGraph)
    blank = np.empty(img.shape)
    splitMergeSeg = splitAndMerge(img, blank, threshold)
    #splitSeg = split(img, np.empty(img.shape), threshold)
    return splitMergeSeg

def splitAndMerge(img, blank, threshold):
    rStack = []
    rStack.append([img, blank, []])
    while len(rStack) != 0:
        pop = rStack.pop()
        halfw = pop[0].shape[1]//2
        halfh = pop[1].shape[0]//2
        print("****************************")
        for i in rStack:
            print(i[0].shape, i[1].shape, i[2])
        print("____________________________")
        if flatTest(pop[0], threshold):
            x,y = pop[0].shape
            avgVal = np.mean(pop[0])
            for i in range(x):
                for j in range(y):
                    blank[i][j] = avgVal
            if len(rStack) >= 1:
                rStack[len(rStack)-1][2].append(avgVal)
        elif len(pop[2]) == 0:
            rStack.append(pop)
            popVal = pop[2]
            tLFrame = [pop[0][:halfh, :halfw], pop[1][:halfh, :halfw], popVal]
            rStack.append(tLFrame)
        elif len(pop[2]) == 1:
            rStack.append(pop)
            popVal = pop[2]
            tRFrame = [pop[0][:halfh, halfw:], pop[1][:halfh, halfw:], popVal]
            rStack.append(tRFrame)
        elif len(pop[2]) == 2:
            rStack.append(pop)
            popVal = pop[2]
            bLFrame = [pop[0][halfh:, :halfw], pop[1][halfh:, :halfw], popVal]
            rStack.append(bLFrame)
        elif len(pop[2]) == 3:
            rStack.append(pop)
            popVal = pop[2]
            bRFrame = [pop[0][halfh:, halfw:], pop[1][halfh:, halfw:], popVal]
            rStack.append(bRFrame)
    return blank

def flatTest(array, threshold):
    x, y = array.shape
    mean = []
    for j in range(x):
        for k in range(y):
            mean.append(array[j,k])
    if len(mean) <= 8 or stat.variance(mean, None) <= threshold[0]:
        return True
    else:
        return False

def eraseEdge(edges, bounds, axis=0):
    print("erasingEdge")
    x,y = edges.shape
    i, j = bounds[0]
    if axis == 1:
        while edges[i-1][j] != 120:
            edges[i][j] = 0
            if i+1 < x:
                edges[i+1][j] = 0
            j += 1
    if axis == 0:
        while edges[i][j-1] != 120:
            edges[i][j] = 0
            if j+1 < y:
                edges[i][j+1] = 0
            i += 1
    return edges

if __name__ == '__main__':

    img = Image.open("22093.jpg")
    array = regionSplitAndMerge(img, (500, img.shape))
    splitAndMerge = array
    newImage = Image.fromarray(splitAndMerge)
    newImage.show()
