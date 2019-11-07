import numpy as np

def euclideanDistance(x1,y1,x2,y2):
    return ((x1-x2)**2+(y1-y2)**2)**(.5)

def createListOfEdgePixels(segmentation):
    out = []
    height,length = segmentation.shape
    for i in range(length):
        for j in range(height):
            for x,y in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]:
                if (x>=0 and y>=0 and x<length and y<height):
                    if (segmentation[y,x] != segmentation[j,i]):
                        out.append((i,j))
                        break
    return out

def findNearestEdgePixel(pixel, edges):
    closest = None
    bestDistance = float("inf")
    for edge in edges:
        distance = euclideanDistance(pixel[0],pixel[1],edge[0],edge[1])
        if distance < bestDistance:
            bestDistance = distance
            closest = edge
    return closest, bestDistance

def bde(segmentation,groundtruth):
    height,length = segmentation.shape
    segEdges = createListOfEdgePixels(segmentation)
    truEdges = createListOfEdgePixels(groundtruth)
    totalDistance = 0
    for edge in segEdges:
        _, distance = findNearestEdgePixel(edge, truEdges)
        totalDistance += distance
    for edge in truEdges:
        _, distance = findNearestEdgePixel(edge, segEdges)
        totalDistance += distance
    return totalDistance / (length*height*2)

if __name__ == "__main__":
    testarray1 = [[1,1,1,1,1,0,0,0],
                  [1,1,1,1,0,0,0,0],
                  [1,1,1,0,0,0,0,0],
                  [1,1,0,0,0,0,0,0]]

    testarray2 = [[0,0,0,1,1,1,1,1],
                  [0,0,0,0,1,1,1,1],
                  [0,0,0,0,0,1,1,1],
                  [0,0,0,0,0,0,1,1]]
    print(bde(testarray1, testarray2))
