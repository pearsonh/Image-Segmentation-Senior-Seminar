from PIL import Image
import numpy as np
from collections import deque




def naive_watershed(image):
    #Creates a greyscale version of the input image
    imageGrey = image.convert(mode="L")
    imageGrey = np.array(imageGrey.getdata()).reshape(imageGrey.size[0], imageGrey.size[1], 1)
    width, height, _  = imageGrey.shape
    getNeighbors = lambda x,y: [i for i in [(x,y+1),(x,y-1),(x+1,y),(x-1,y)] if (i[0] >= 0 and i[1] >= 0 and i[0] < width and i[1] < height)]
    curLabel = 2 # 0 is unlabelled. 1 is WALL
    pix = [deque() for i in range(256)]
    for x in range(width):
        for y in range(height):
            pix[int(imageGrey[(x,y)])].append((x,y))
    labelImage = np.zeros(shape=(width,height))
    returnImage = np.zeros(shape=(width,height))
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
            #print("we")
            for pixel in pixList:
                #print(pixel, label)
                labelImage[pixel] = label
                if label == 1:
                    returnImage[pixel] = 1
    print(returnImage)
    print(labelImage)
    print(imageGrey)
    return returnImage

def getClumpAndLowNeighbors(img,pix,neighb):
    print()
    print(pix)
    val = img[pix]
    pixlist = [pix]
    neighbors = []
    stack = neighb(*pix)
    while len(stack) > 0:
        print("AUDITION:")
        nextpix = stack.pop()
        print(nextpix)
        if ((val == img[nextpix]) and (nextpix not in pixlist)):
            pixlist.append(nextpix)
            for i in neighb(*nextpix):
                stack.append(i)
        elif ((val > img[nextpix]) and (nextpix not in neighbors)):
            neighbors.append(nextpix)
    print("and the gang:")
    print(pixlist)
    print("and their neighbors")
    print(neighbors)
    return pixlist, neighbors

if __name__ == "__main__":
    img = Image.open("test.jpg")
    img = Image.fromarray(naive_watershed(img)*220)
    img.show()
