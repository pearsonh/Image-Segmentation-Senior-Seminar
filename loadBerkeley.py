import numpy as np
from PIL import Image

def loadSegmentation(filename):
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

if __name__ == "__main__":
    img = Image.fromarray(loadSegmentation("22093.seg")*20)
    img.show()
