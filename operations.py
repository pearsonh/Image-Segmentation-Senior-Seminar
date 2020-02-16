from thresholding import baseline_thresholding as bt
from PIL import Image
import numpy as np
import os
from collections import Counter

from eval import bde
import parser
'''WARNING: THIS SCRIPT IS BUILT TO BE RUN ON TH CMC 307 LAB COMPUTER
AND WILL NOT WORK ON OTHERS WITHOUT CHANGING THE DIRECTORIES'''
def import_berkeley(filename):
    '''Given the path to a file containing a Berkeley encoding of a segmentation,
    returns the numpy version of that segmentation'''
    linelist = None
    with open(filename) as segFile:
        linelist = [line for line in segFile]
    i = 0
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


def experiment():
    location_segments = "C:\\Users\\brydu\\OneDrive\\Desktop\\Comps\\Image-Segmentation-Senior-Seminar\\threshold_segments\\"
    location_truths = "C:\\Users\\brydu\\OneDrive\\Desktop\\Comps\\Image-Segmentation-Senior-Seminar\\threshold_segments_truths\\"

    bde_accuracy_averages = {}
    files = os.listdir(location_truths)
    for file in files:
        print("EXECUTING " + file)
        segmentation = "segmentation" + file[:-4] + ".jpg"
        seg = Image.open(location_segments + segmentation)
        print("IMAGE OPENED")
        seg = np.array(seg)
        truth = import_berkeley(location_truths + file)
        print("TRUTH IMPORTED")
        accuracy = bde(seg, truth)
        print(accuracy)
        print("ACCURACY DISCOVERED")
        bde_accuracy_averages[file] = accuracy
    print(bde_accuracy_averages)

def segment_all_images():
    location = "BSDS300/images"
    print(os.path.expanduser(location))
    print("FOUND IT")
    files = os.listdir("BSDS300/images")
    
    for file in files:
        img = Image.open(location + '/' + file)
        temp = bt(img)
        img = Image.fromarray(temp * 100)
        img = img.convert("L")
        img.save("thresholding_segmentations/" + file)

if __name__ == "__main__":
    segment_all_images()