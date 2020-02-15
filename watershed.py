from PIL import Image
from PIL import ImageFilter
import numpy as np
from collections import deque
from random import randrange

def naive_watershed(image,depththreshold,blur=None):
    #blurs input image
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur)) #blur
    #Creates a greyscale version of the input image
    imageGrey = image.convert(mode="L")
    imageGrey = np.array(imageGrey.getdata()).reshape(imageGrey.size[1],imageGrey.size[0])
    #imageGrey = imageGrey - imageGrey%18 #round
    height, width = imageGrey.shape
    getNeighbors = lambda x,y: [i for i in [(x,y+1),(x,y-1),(x+1,y),(x-1,y)] if (i[0] >= 0 and i[1] >= 0 and i[0] < height and i[1] < width)]
    curLabel = 2 # 0 is unlabelled. 1 is WALL
    pix = [deque() for i in range(256+5)]
    for x in range(height):
        for y in range(width):
            pix[int(imageGrey[(x,y)])].append((x,y))
    labelImage = np.zeros(shape=(height,width))
    returnImage = np.zeros(shape=(height,width,3))
    listofmins = ["???","???"]
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


    for x in range(height):
        for y in range(width):
            pixel = (x,y)
            if labelImage[pixel] == 1:
                pixelsToDye, val = getClumpForDyeing(imageGrey,pixel,getNeighbors,labelImage)
                for pixel in pixelsToDye:
                    labelImage[pixel] = val

    colorkey = np.zeros((curLabel,3))
    for i in range(curLabel):
        colorkey[i] = (randrange(256),randrange(256),randrange(256))
    for x in range(height):
        for y in range(width):
            pixel = (x,y)
            returnImage[x,y] = colorkey[int(labelImage[pixel])]
    return returnImage

def watershed_with_wolf(image,depththreshold,blur=None):
    #blurs input image
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur)) #blur
    #Creates a greyscale version of the input image
    imageGrey = image.convert(mode="L")
    imageGrey = np.array(imageGrey.getdata()).reshape(imageGrey.size[1],imageGrey.size[0])
    #imageGrey = imageGrey - imageGrey%18 #round
    height, width = imageGrey.shape
    getNeighbors = lambda x,y: [i for i in [(x,y+1),(x,y-1),(x+1,y),(x-1,y)] if (i[0] >= 0 and i[1] >= 0 and i[0] < height and i[1] < width)]
    curLabel = 2 # 0 is unlabelled. 1 is WALL
    pix = [deque() for i in range(256+5)]
    for x in range(height):
        for y in range(width):
            pix[int(imageGrey[(x,y)])].append((x,y))
    labelImage = np.zeros(shape=(height,width))
    returnImage = np.zeros(shape=(height,width,3))
    listofmins = ["???","???"]
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
                listofmins.append(imageGrey[nextpix])
            else:
                label = labels[0]
                for curLab in labels:
                    if curLab != label:
                        label = 1
                        break
            if label != 1:
                for pixel in pixList:
                    labelImage[pixel] = label
            else:
                deepEnough = []
                notDeepEnough = []
                nextpixval = imageGrey[nextpix]
                #sort through all neighbors
                #separate out the ones that don't pass muster, from the ones that do
                for neighbor in labels:
                    if neighbor == 0:
                        continue
                    if (nextpixval - listofmins[int(neighbor)] > depththreshold):
                        deepEnough.append(neighbor)
                    else:
                        notDeepEnough.append(neighbor)
                #if there are none that don't, proceed
                if (len(notDeepEnough) == 0):
                    for pixel in pixList:
                        labelImage[pixel] = 1
                    continue
                else:
                    if len(deepEnough) > 0:
                        target = deepEnough[0]
                    else:
                        target = notDeepEnough[0]
                #pick a target label from the ones that do (if there are none, from the ones that don't)
                #assign all the ones that don't, to that label, using a replace function arr[arr > 255] = x
                    for repl in notDeepEnough:
                        labelImage[np.absolute(labelImage - repl) < 0.1] = target
                #and assign that value, for nextpix itself. thanx
                    for pixel in pixList:
                        labelImage[pixel] = target
    
    for x in range(height):
        for y in range(width):
            pixel = (x,y)
            if labelImage[pixel] == 1:
                pixelsToDye, val = getClumpForDyeing(imageGrey,pixel,getNeighbors,labelImage)
                for pixel in pixelsToDye:
                    labelImage[pixel] = val
    
    colorkey = np.zeros((curLabel,3))
    for i in range(curLabel):
        colorkey[i] = (randrange(256),randrange(256),randrange(256))
    for x in range(height):
        for y in range(width):
            pixel = (x,y)
            returnImage[x,y] = colorkey[int(labelImage[pixel])]
    return returnImage

def getClumpForDyeing(img,pix,neighb,labelimg):
    val = int(img[pix])
    pixlist = [pix]
    stack = neighb(*pix)
    while len(stack) > 0:
        nextpix = stack.pop()
        if ((val == int(img[nextpix])) and (nextpix not in pixlist)):
            pixlist.append(nextpix)
            for i in neighb(*nextpix):
                stack.append(i)
        elif (nextpix not in pixlist) and (labelimg[nextpix] != 1):
            dye = labelimg[nextpix]
    return pixlist, dye

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
    img = Image.fromarray(naive_watershed(img,30).astype('uint8'))
    img.show()
