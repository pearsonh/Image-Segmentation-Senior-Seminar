# this file contains functions that load ground truth files with weizmann or berkeley encodings into numpy arrays
import numpy as np
from PIL import Image

def import_berkeley(filename):
    '''Given the path to a file containing a Berkeley encoding of a segmentation,
    returns the numpy version of that segmentation'''
    linelist = None
    with open(filename) as segFile:
        linelist = [line for line in segFile]
    i = 0;
    width = None
    height = None
    start_line = None
    while not (width and height and start_line):
        if linelist[i].startswith("width"):
            width = int(linelist[i].split(" ")[1])
        if linelist[i].startswith("height"):
            height = int(linelist[i].split(" ")[1])
        if linelist[i].startswith("data"):
            start_line = i+1
        i += 1
    segarray = -np.ones((height,width))
    for line in range(start_line,len(linelist)):
        seg, row, start, end = linelist[line].split(" ")
        for i in range(int(start),int(end)+1):
            segarray[int(row),i] = int(seg)
    return segarray

def import_weizmann_1(filename):
    '''Given the path to a file containing a Weizmann encoding of a segmentation,
    returns the numpy version of that segmentation'''
    image = Image.open(filename)
    image_matrix=np.array(image)
    for i in range(image.height):
        for j in range(image.width):
            rgb=image_matrix[i][j]
            if (rgb[0] != rgb[1]) or (rgb[1] != rgb[2]) or (rgb[2] != rgb[0]):
                print("seg")
                image_matrix[i][j]=1
            else:
                image_matrix[i][j]=0
    print(image_matrix)

def display_segmentation(image,number_of_segmentations):
    '''given a numpy array with a segmentation, displays that segmentation'''
    img = Image.fromarray(image*255/number_of_segmentations)
    img.show()

if __name__ == "__main__":
    image = import_berkeley("22093.seg")
    display_segmentation(image,12)
