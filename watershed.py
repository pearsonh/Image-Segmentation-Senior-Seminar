from PIL import Image
from PIL import ImageFilter
import numpy as np
from collections import deque

def naive_watershed(image):
    #Creates a greyscale version of the input image
    imageGrey = image.convert(mode="L")
    imageGrey = np.array(imageGrey.getdata()).reshape(imageGrey.size[1],imageGrey.size[0])
    #imageGrey = np.round(imageGrey,-1)
    height, width = imageGrey.shape
    getNeighbors = lambda x,y: [i for i in [(x,y+1),(x,y-1),(x+1,y),(x-1,y)] if (i[0] >= 0 and i[1] >= 0 and i[0] < height and i[1] < width)]
    curLabel = 2 # 0 is unlabelled. 1 is WALL
    pix = [deque() for i in range(256+5)]
    for x in range(height):
        for y in range(width):
            pix[int(imageGrey[(x,y)])].append((x,y))
    labelImage = np.zeros(shape=(height,width))
    returnImage = np.zeros(shape=(height,width))
    for val in range(256):
        while True:
            try:
                nextpix = pix[val].pop()
            except IndexError:
                break
            if (labelImage[nextpix] != 0):
                continue
            pixList, neighbors = getClumpAndLowNeighbors(imageGrey,nextpix,getNeighbors)
            labels = []
            for neighbor in neighbors:
                if labelImage[neighbor] != 1: #not one of those WALLs
                    labels.append(labelImage[neighbor])
                    if labelImage[neighbor] == 0:
                        raise Exception
            if len(labels) == 0:
                label = curLabel
                curLabel = curLabel + 1
            else:
                label = labels[0]
                for curLab in labels:
                    if curLab != label:
                        label = 1
                        break
            for pixel in pixList:
                labelImage[pixel] = label
                if label == 1:
                    returnImage[pixel] = 1
    return returnImage

def getClumpAndLowNeighbors(img,pix,neighb):
    val = img[pix]
    pixlist = [pix]
    neighbors = []
    stack = neighb(*pix)
    while len(stack) > 0:
        nextpix = stack.pop()
        if ((val == img[nextpix]) and (nextpix not in pixlist)):
            pixlist.append(nextpix)
            for i in neighb(*nextpix):
                stack.append(i)
        elif ((val > img[nextpix]) and (nextpix not in neighbors)):
            neighbors.append(nextpix)
    return pixlist, neighbors

if __name__ == "__main__":
    img = Image.open("22093.jpg")
    img = img.filter(ImageFilter.GaussianBlur(2))
    img = Image.fromarray(naive_watershed(img)*220)
    img.show()
