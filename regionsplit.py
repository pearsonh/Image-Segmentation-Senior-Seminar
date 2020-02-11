from PIL import Image
import numpy as np
import statistics as stat
import math
from scipy import spatial as sc
'''class regionSplitTree:
    def __init__(self, root):
        self.root = root
        self.size = 0

    def split(self, threshold):
        return self.root.split(threshold)

    def merge(self, threshold):
        return self.root.merge(threshold)

class quadNode:
    def __init__(self, data, parent=None, regions=[]):
        self.data = data
        self.parent = parent
        self.regions = regions

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

    def split(self):
            self.regions.append(quadNode([(self.data[0][0]//2, self.data[0][1]//2), self.data[1]], self))
            self.regions.append(quadNode([(self.data[0][0]//2, self.data[0][1]//2), (self.data[0][0]//2, self.data[1][1])], self))
            self.regions.append(quadNode([(self.data[0][0]//2, self.data[0][1]//2), (self.data[1][0], self.data[0][1]//2)], self))
            self.regions.append(quadNode([(self.data[0][0]//2, self.data[0][1]//2), (self.data[0][0]//2, self.data[0][1]//2)], self))
            #print(len(self.regions))
            return

'''
def regionSplitAndMerge(image, threshold):
    imageGrey = image.convert(mode='L')
    '''imgTree = regionSplitTree(quadNode(np.array(imageGrey).astype('int')))
    print(imgTree.root)
    imgTree.split(threshold)
    imgTree.merge(threshold)'''
    img = np.array(imageGrey).astype("int")
    #adjGraph = quadNode([img.shape, [0,0]])
    #print(adjGraph.regions)
    #print(adjGraph)
    splitMergeSeg = splitAndMerge(img, np.empty(img.shape), threshold)
    splitSeg = split(img, np.empty(img.shape), threshold)
    '''regionSegmentation = merge(img, splitSeg, adjGraph, threshold)'''
    #splitMergeSegTwoColor = twoColor(splitMergeSeg)
    return (splitMergeSeg, splitSeg)

def splitAndMerge(img, blank, threshold):
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
        print("Merging")
        if flatTest(np.concatenate([tL, tR], axis=1), threshold):
            print("topMerge")
            blank[:len(blank)//2, :len(blank)//2] = np.full(tL.shape, np.mean(np.concatenate([tL, tR], axis=1)))
            blank[:len(blank)//2, len(blank)//2:] = np.full(tR.shape, np.mean(np.concatenate([tL, tR], axis=1)))
            #blank[:len(blank)//2, :len(blank)] = np.full(np.concatenate([tL, tR], axis=1).shape, np.mean(np.concatenate([tL, tR], axis=1)))
        if flatTest(np.concatenate([bL, bR], axis=1), threshold):
            print("bottomMerge")
            blank[len(blank)//2:, :len(blank)//2] = np.full(bL.shape, np.mean(np.concatenate([bL, bR], axis=1)))
            blank[len(blank)//2:, len(blank)//2:] = np.full(bR.shape, np.mean(np.concatenate([bL, bR], axis=1)))
            #blank[len(blank)//2:, :len(blank)] = np.full(np.concatenate([bL, bR], axis=1).shape, np.mean(np.concatenate([bL, bR], axis=1)))
        if flatTest(np.concatenate([tL, bL], axis=0), threshold):
            print("leftMerge")
            blank[:len(blank)//2, :len(blank)//2] = np.full(tL.shape, np.mean(np.concatenate([tL, bL], axis=0)))
            blank[len(blank)//2:, :len(blank)//2] = np.full(bL.shape, np.mean(np.concatenate([tL, bL], axis=0)))
            #blank[:len(blank), :len(blank)//2] = np.full(np.concatenate([tL, bL], axis=0).shape, np.mean(np.concatenate([tL, bL], axis=0)))
        if flatTest(np.concatenate([tR, bR], axis=0), threshold):
            print("rightMerge")
            blank[:len(blank)//2, len(blank)//2:] = np.full(tR.shape, np.mean(np.concatenate([tR, bR], axis=0)))
            blank[len(blank)//2:, len(blank)//2:] = np.full(bR.shape, np.mean(np.concatenate([tR, bR], axis=0)))
            #blank[:len(blank), len(blank)//2:] = np.full(np.concatenate([tR, bR], axis=0).shape, np.mean(np.concatenate([tR, bR], axis=0)))
        splitAndMerge(tL, blank[:len(blank)//2, :len(blank)//2], threshold)
        splitAndMerge(tR, blank[:len(blank)//2, len(blank)//2:],  threshold)
        splitAndMerge(bL, blank[len(blank)//2:, :len(blank)//2],  threshold)
        splitAndMerge(bR, blank[len(blank)//2:, len(blank)//2:],  threshold)
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
        '''if flatTest(np.concatenate([tL, tR], axis=1), threshold):
            blank[:len(blank)//2, :len(blank)//2] = np.mean(np.concatenate([tL, tR], axis=1))
            blank[:len(blank)//2, len(blank)//2:] = np.mean(np.concatenate([tL, tR], axis=1))
        if flatTest(np.concatenate([bL, bR], axis=1), threshold):
            blank[len(blank)//2:, :len(blank)//2] = np.mean(np.concatenate([bL, bR], axis=1))
            blank[len(blank)//2:, len(blank)//2:] = np.mean(np.concatenate([bL, bR], axis=1))
        if flatTest(np.concatenate([tL, bL], axis=0), threshold):
            blank[:len(blank)//2, :len(blank)//2] = np.mean(np.concatenate([tL, bL], axis=0))
            blank[len(blank)//2, :len(blank)//2:] = np.mean(np.concatenate([tL, bL], axis=0))
        if flatTest(np.concatenate([tR, bR], axis=0), threshold):
            blank[:len(blank)//2, len(blank)//2:] = np.mean(np.concatenate([tR, bR], axis=0))
            blank[len(blank)//2:, len(blank)//2:] = np.mean(np.concatenate([tR, bR], axis=0))'''
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

def twoColor(img):
    x,y = img.shape
    mean = np.mean(img)
    print(mean)
    for i in range(x):
        for j in range(y):
            if img[i][j] < mean - mean//8:
                img[i][j] = 0
            elif img[i][j] > mean + mean//8:
                img[i][j] = 255
            else:
                print("Middle")
                img[i][j] = mean
    return img


if __name__ == '__main__':
    img = Image.open("22093.jpg")
    array = regionSplitAndMerge(img, 400)
    splitAndMerge = array[0]
    split = array[1]
    newImage = Image.fromarray(splitAndMerge)
    newImage.show()
    splitImage = Image.fromarray(split)
    splitImage.show()
    if newImage == splitImage:
        print("Same Image")

'''for i in range(x):
    for j in range(y):
        if splitArray[i][j] != regionColor:
            edges = findSplit(splitArray, i, j)
            print(edges)
            baseRegion = (image[edges[0]:edges[1], edges[2]:edges[3]], splitArray[edges[0]:edges[1], edges[2]:edges[3]])
            if edges[0] != 1:
                mergeEdges = findSplit(splitArray, edges[0]-1, j)
                print(mergeEdges)
                mergeRegion = (image[mergeEdges[0]:mergeEdges[1], mergeEdges[2]:mergeEdges[3]], splitArray[mergeEdges[0]:mergeEdges[1], mergeEdges[2]:mergeEdges[3]])
                print(str(baseRegion[0].shape) + ':' + str(mergeRegion[0].shape))
                if flatTest(np.concatenate([mergeRegion[0], baseRegion[0]], axis=1), threshold):
                    #print("Repainting")
                    mergeRegion = np.full(mergeRegion[1].shape, regionColor)
            if edges[1] != x-1:
                mergeEdges = findSplit(splitArray, edges[1]+1, j)
                print(mergeEdges)
                mergeRegion = (image[mergeEdges[0]:mergeEdges[1], mergeEdges[2]:mergeEdges[3]], splitArray[mergeEdges[0]:mergeEdges[1], mergeEdges[2]:mergeEdges[3]])
                print(str(baseRegion[0].shape) + ':' + str(mergeRegion[0].shape))
                if flatTest(np.concatenate([mergeRegion[0], baseRegion[0]], axis=0), threshold):
                    #print("Repainting")
                    mergeRegion = np.full(mergeRegion[1].shape, regionColor)
            if edges[2] != 1:
                mergeEdges = findSplit(splitArray, i, edges[2]-1)
                print(mergeEdges)
                mergeRegion = (image[mergeEdges[0]:mergeEdges[1], mergeEdges[2]:mergeEdges[3]], splitArray[mergeEdges[0]:mergeEdges[1], mergeEdges[2]:mergeEdges[3]])
                print(str(baseRegion[0].shape) + ':' + str(mergeRegion[0].shape))
                if flatTest(np.concatenate([mergeRegion[0], baseRegion[0]], axis=1), threshold):
                    #print("Repainting")
                    mergeRegion = np.full(mergeRegion[1].shape, regionColor)
            if edges[3] != y-1:
                mergeEdges = findSplit(splitArray, i, edges[3]+1)
                print(mergeEdges)
                mergeRegion = (image[mergeEdges[0]:mergeEdges[1], mergeEdges[2]:mergeEdges[3]], splitArray[mergeEdges[0]:mergeEdges[1], mergeEdges[2]:mergeEdges[3]])
                print(str(baseRegion[0].shape) + ':' + str(mergeRegion[0].shape))
                if flatTest(np.concatenate([mergeRegion[0], baseRegion[0]], axis=1), threshold):
                    #print("Repainting")
                    mergeRegion = np.full(mergeRegion[1].shape, regionColor)'''
