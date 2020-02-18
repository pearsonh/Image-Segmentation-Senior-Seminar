from PIL import Image
from PIL import ImageFilter
import numpy as np
from collections import deque
from random import randrange

#this function is supposed to implement watershed with merging, however it's too slow to be practical and may have bugs, so don't use it
def watershed_with_merging(image,mergethreshold,blur=None):
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
    regionSizes = [0,0]
    regionTotalColors = [0,0]
    regionConnections = []
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
                regionSizes.append(0)
                regionTotalColors.append(0)
                label = curLabel
                curLabel = curLabel + 1
            else:
                uniqueLabels = [0]
                for curLab in labels:
                    if int(curLab) not in uniqueLabels:
                        uniqueLabels.append(int(curLab))
                uniqueLabels = uniqueLabels[1:]
                if len(uniqueLabels) == 0:
                    raise Exception
                elif len(uniqueLabels) == 1:
                    label = uniqueLabels[0]
                else:
                    for i in range(len(uniqueLabels)):
                        for j in range(i+1,len(uniqueLabels)):
                            regionConnections.append([uniqueLabels[i],uniqueLabels[j]])
            for pixel in pixList:
                labelImage[pixel] = label
                regionSizes[int(label)] += 1
                regionTotalColors[int(label)] += imageGrey[pixel]

    #iterate through all pairs. apply distance metric, and merge if necessary
    removedSet = {1}
    mergedOnce = True
    print("begin merging")
    while mergedOnce:
        print("merge")
        mergedOnce = False
        for reg1, reg2 in regionConnections:
            if (reg1 not in removedSet) and (reg2 not in removedSet):
                print(reg2, regionSizes[reg2], reg1, regionSizes[reg1])
                if abs((regionTotalColors[reg1]/regionSizes[reg1]) - (regionTotalColors[reg2]/regionSizes[reg2])) < mergethreshold:
                    removedSet.add(reg2)
                    labelImage[np.absolute(labelImage - reg2) < 0.1] = reg1
                    regionTotalColors[reg1] += regionTotalColors[reg2]
                    regionSizes[reg1] += regionSizes[reg2]
                    mergedOnce = True
                    for newreg1, newreg2 in regionConnections:
                        if newreg1 == reg2:
                            regionConnections.append([reg1, newreg1])
                        if newreg2 == reg2:
                            regionConnections.append([reg1, newreg2])
        regionConnections = [i for i in regionConnections if ((i[0] not in removedSet) and (i[1] not in removedSet))]
    print("finished once")


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
    return returnImage.astype('uint8')

#just a plain old watershed segmentation, nothing unique about it
def naive_watershed(image,blur=None):
    #blurs input image
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur)) #blur
    #Creates a greyscale version of the input image
    imageGrey = image.convert(mode="L")
    imageGrey = np.array(imageGrey.getdata()).reshape(imageGrey.size[1],imageGrey.size[0])
    #imageGrey = imageGrey - imageGrey%18 #round, now deprecated because I didn't think it was that good
    height, width = imageGrey.shape
    getNeighbors = lambda x,y: [i for i in [(x,y+1),(x,y-1),(x+1,y),(x-1,y)] if (i[0] >= 0 and i[1] >= 0 and i[0] < height and i[1] < width)] #this defines a function used to get the neighbors of a pixel
    curLabel = 2 # 0 is unlabelled. 1 is WALL
    pix = [deque() for i in range(256+5)] #priority queue of pixels by color
    for x in range(height):
        for y in range(width):
            pix[int(imageGrey[(x,y)])].append((x,y))
    labelImage = np.zeros(shape=(height,width)) #0 is unlabelled. 1 is "WALL" or border between region
    returnImage = np.zeros(shape=(height,width,3))
    listofmins = ["???","???"]
    for val in range(256):
        while True:
            try:
                nextpix = pix[val].pop()
            except IndexError:
                break  #if the current queue is out of pixels, we break out of the while loop
            if (labelImage[nextpix] != 0):
                continue
            pixList, neighbors = getClumpAndLowNeighbors(imageGrey,nextpix,getNeighbors) #gets clump of pixels and border pixels
            labels = []
            for neighbor in neighbors:
                if labelImage[neighbor] != 1: #not one of those WALLs
                    labels.append(labelImage[neighbor])
                    if labelImage[neighbor] == 0: #none should be unlabelled
                        raise Exception
            if len(labels) == 0: #no labelled borders, start a new region
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


    for x in range(height): #gets all border pixels and forces them to pick a side
        for y in range(width):
            pixel = (x,y)
            if labelImage[pixel] == 1:
                pixelsToDye, val = getClumpForDyeing(imageGrey,pixel,getNeighbors,labelImage)
                for pixel in pixelsToDye:
                    labelImage[pixel] = val

    colorkey = np.zeros((curLabel,3)) #colors the pixels based on the region they're a part of
    for i in range(curLabel):
        colorkey[i] = (randrange(256),randrange(256),randrange(256))
    for x in range(height):
        for y in range(width):
            pixel = (x,y)
            returnImage[x,y] = colorkey[int(labelImage[pixel])]
    return returnImage.astype('uint8')

def watershed_with_wolf(image,depththreshold,blur=None):
    #essentially, first executes watershed transform (collecting a bit of extra data), dynamically using wolf pruning in the process, then colors the image and sends it back
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
            else: #if label was equal to one, checks if the two regions bordering THIS region can be combined into a big region that's better
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
    
    #colors in image, finally
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
    return returnImage.astype('uint8')

#gets a clump of connected pixels with the same value, and the label of a region that borders them (so that they can all be converted to that label
def getClumpForDyeing(img,pix,neighb,labelimg):
    val = int(img[pix])
    pixlist = [pix]
    stack = neighb(*pix) #stack for DFS
    while len(stack) > 0:
        nextpix = stack.pop()
        if ((val == int(img[nextpix])) and (nextpix not in pixlist)):
            pixlist.append(nextpix)
            for i in neighb(*nextpix):
                stack.append(i)
        elif (nextpix not in pixlist) and (labelimg[nextpix] != 1):
            dye = labelimg[nextpix]
    return pixlist, dye

#gets a clump of connected pixels with the same value, as well as their neighbors. give it an image, seed pixel, and a function for getting neighbors
def getClumpAndLowNeighbors(img,pix,neighb):
    val = img[pix]
    pixlist = [pix]
    neighbors = []
    stack = neighb(*pix) #uses stack for depth first search
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
    img = Image.fromarray(watershed_with_wolf(img,30,blur=2))
    img.show()
