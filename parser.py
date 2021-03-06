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
def import_weizmann(filename):
    '''Given the path to a file containing a Weizmann encoding of a segmentation,
    returns the numpy version of that segmentation'''
    image = Image.open(filename)
    image_matrix=np.array(image)
    segmentedmin = np.argmin(image_matrix, axis=2)
    segmentedmax = np.argmax(image_matrix, axis=2)
    segmented=np.add(segmentedmin, segmentedmax)/2
    return segmented

def display_segmentation(image,number_of_segmentations):
    '''given a numpy array with a segmentation, displays that segmentation'''
    img = Image.fromarray(image*255/number_of_segmentations)
    img.show()

if __name__ == "__main__":
    image = import_berkeley("22093.seg")
    display_segmentation(image,12)
