from PIL import Image
import numpy as np
import statistics as stat
import math
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

def splitAndMerge(img, blank, edges=[], threshold):
    if flatTest(img, threshold):
        x,y = img.shape
        avgVal = np.mean(img)
        for i in range(x):
            for j in range(y):
                #blank[i][j] = regionValue * 20
                blank[i][j] = avgVal
                if i == 0 or j == 0:
                    #edges[i][j] = 120
    else:
        tL = img[:len(img)//2, :len(img)//2]
        tR = img[:len(img)//2, len(img)//2:]
        bL = img[len(img)//2:, :len(img)//2]
        bR = img[len(img)//2:, len(img)//2:]
        '''splitAndMerge(tL, blank[:len(blank)//2, :len(blank)//2], edges[:len(edges)//2, :len(edges)//2], threshold)
        splitAndMerge(tR, blank[:len(blank)//2, len(blank)//2:], edges[:len(edges)//2, len(edges)//2:],  threshold)
        splitAndMerge(bL, blank[len(blank)//2:, :len(blank)//2], edges[len(edges)//2:, :len(edges)//2], threshold)
        splitAndMerge(bR, blank[len(blank)//2:, len(blank)//2:], edges[len(edges)//2:, len(edges)//2:], threshold)'''
        splitAndMerge(tL, blank[:len(blank)//2, :len(blank)//2], threshold)
        splitAndMerge(tR, blank[:len(blank)//2, len(blank)//2:], threshold)
        splitAndMerge(bL, blank[len(blank)//2:, :len(blank)//2], threshold)
        splitAndMerge(bR, blank[len(blank)//2:, len(blank)//2:], threshold)
        if flatTest(np.concatenate([tL, tR], axis=1), threshold):
            blank[:len(blank)//2, :len(blank)//2] = np.full(tL.shape, np.mean(np.concatenate([tL, tR], axis=1)))
            blank[:len(blank)//2, len(blank)//2:] = np.full(tR.shape, np.mean(np.concatenate([tL, tR], axis=1)))
            '''if np.concatenate([tL, tR]).shape[0] > 4:
                eraseEdge(edges, ((len(blank)//2, 0), (len(blank)//2, len(blank)//2)), axis=1)'''
            #blank[:len(blank)//2, :len(blank)] = np.full(np.concatenate([tL, tR], axis=1).shape, np.mean(np.concatenate([tL, tR], axis=1)))
        if flatTest(np.concatenate([bL, bR], axis=1), threshold):
            blank[len(blank)//2:, :len(blank)//2] = np.full(bL.shape, np.mean(np.concatenate([bL, bR], axis=1)))
            blank[len(blank)//2:, len(blank)//2:] = np.full(bR.shape, np.mean(np.concatenate([bL, bR], axis=1)))
            '''if np.concatenate([bL, bR]).shape[0] > 4:
                eraseEdge(edges, ((len(blank)//2, len(blank)//2), (len(blank)//2, len(blank))), axis=1)'''
            #blank[len(blank)//2:, :len(blank)] = np.full(np.concatenate([bL, bR], axis=1).shape, np.mean(np.concatenate([bL, bR], axis=1)))
        if flatTest(np.concatenate([tR, bR], axis=0), threshold):
            blank[:len(blank)//2, len(blank)//2:] = np.full(tR.shape, np.mean(np.concatenate([tR, bR], axis=0)))
            blank[len(blank)//2:, len(blank)//2:] = np.full(bR.shape, np.mean(np.concatenate([tR, bR], axis=0)))
            '''if np.concatenate([tR, bR]).shape[1] > 4:
                eraseEdge(edges, ((0, len(blank)//2), (len(blank)//2, len(blank)//2)), axis=0)'''
            #blank[:len(blank), len(blank)//2:] = np.full(np.concatenate([tR, bR], axis=0).shape, np.mean(np.concatenate([tR, bR], axis=0)))
        if flatTest(np.concatenate([tL, bL], axis=0), threshold):
            blank[:len(blank)//2, :len(blank)//2] = np.full(tL.shape, np.mean(np.concatenate([tL, bL], axis=0)))
            blank[len(blank)//2:, :len(blank)//2] = np.full(bL.shape, np.mean(np.concatenate([tL, bL], axis=0)))
            '''if np.concatenate([tL, bL]).shape[1] > 4:
                eraseEdge(edges, ((len(blank)//2, len(blank)//2), (len(blank), len(blank)//2)), axis=0)'''
            #blank[:len(blank), :len(blank)//2] = np.full(np.concatenate([tL, bL], axis=0).shape, np.mean(np.concatenate([tL, bL], axis=0)))


        return blank

def split(img, blank, threshold):
    if flatTest(img, threshold):
        x,y = img.shape
        avgVal = np.mean(img)
        for i in range(x):
            for j in range(y):
                #blank[i][j] = regionValue * 20
                blank[i][j] = avgVal
    else:
        #graph.split()
        tL = img[:len(img)//2, :len(img)//2]
        tR = img[:len(img)//2, len(img)//2:]
        bL = img[len(img)//2:, :len(img)//2]
        bR = img[len(img)//2:, len(img)//2:]
        split(tL, blank[:len(blank)//2, :len(blank)//2], threshold)
        split(tR, blank[:len(blank)//2, len(blank)//2:],  threshold)
        split(bL, blank[len(blank)//2:, :len(blank)//2],  threshold)
        split(bR, blank[len(blank)//2:, len(blank)//2:],  threshold)
        return blank

def flatTest(array, threshold):
    x, y = array.shape
    mean = []
    for j in range(x):
        for k in range(y):
            mean.append(array[j,k])
    if len(mean) <= 4 or stat.variance(mean, None) <= threshold:
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
    array = regionSplitAndMerge(img, 500)
    splitAndMerge = array
    newImage = Image.fromarray(splitAndMerge)
    if newImage == splitImage:
        print("Same Image")
