from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu, threshold_multiotsu
import os
from collections import Counter

from eval import bde
import parser



def otsu_greyspace_thresholding(image):
    #Creates a greyscale version of the input image
    imageGrey = image.convert(mode="L")

    x, y = imageGrey.size
    #Gets a threshold for the image
    threshold = get_histogram_threshold(imageGrey)[0]
    print(threshold)
    returnImage = np.zeros(shape=(y,x))
    segments = len(threshold)
    #Maps 1's and 0's based on whether the pixel values are above or below the threshold.
    if len(threshold) is 1:
        for j in range(x):
            for i in range(y):
                returnImage[i,j] = 1 if imageGrey.getpixel((j,i)) > threshold else 0
    else:
        for j in range(x):
            for i in range(y):
                if imageGrey.getpixel((j,i)) < threshold[0]:
                    returnImage[i,j] = 0
                if imageGrey.getpixel((j,i)) > threshold[0]:
                    returnImage[i,j] = 1
                if imageGrey.getpixel((j,i)) > threshold[1]:
                    returnImage[i,j] = 2

    return returnImage, segments

def otsu_multi_greyspace_thresholding(image, segments):
    imageGrey = image.convert(mode="L")

    x, y = imageGrey.size
    print(imageGrey.histogram())
    #Gets a threshold for the image
    thresholds = threshold_multiotsu(np.array(image), segments)
    print(thresholds)
    returnImage = np.zeros(shape=(y,x))
    #Maps 1's and 0's based on whether the pixel values are above or below the threshold.
    for j in range(x):
        for i in range(y):
            segment = 0
            for p in range(segments - 1):
                if imageGrey.getpixel((j,i)) > thresholds[p]:
                    segment = p + 1
            returnImage[i,j] = segment

    returnImage = smoothy(returnImage)
    unique_elements, counts_elements = np.unique(returnImage, return_counts=True)
    print("Frequency of unique values of the said array:")
    print(np.asarray((unique_elements, counts_elements)))
    return returnImage, segments

def smoothy(image_array):
    x, y = image_array.shape
    for i in range(1,x-1):
        for j in range(1,y-1):
            neighbor_segments = check_neighbors(image_array, i, j)
            b = Counter(neighbor_segments)
            most_common_segment = b.most_common()
            neighbor_segments = (neighbor_segments == most_common_segment)
            print(most_common_segment)
            print(neighbor_segments)
            if (neighbor_segments == most_common_segment).sum() > 5:
                image_array[i,j] = most_common_segment
    return image_array


def check_neighbors(image_array,x,y):
    neighbors = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1),(1,1)]
    segment = image_array[x,y]
    neighbor_segments = []
    for neighbor in neighbors:
        neighbor_segments.append(image_array[x + neighbor[0], y + neighbor[1]])
    neighbor_segments = np.array(neighbor_segments)
    return np.array(neighbor_segments)


def get_histogram_threshold(image):
    a = np.array(image.histogram())
    return np.where(a == threshold_otsu(a))






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
    location = "C:/Users/brydu/OneDrive/Desktop/Comps/Image-Segmentation-Senior-Seminar/BSDS300-images/BSDS300/images/test/"
    files = os.listdir(location)
    for file in files:
        img = Image.open(location + file)
        temp, segments = otsu_greyspace_thresholding(img)
        img = Image.fromarray(temp * 50)
        img = img.convert("L")
        img.save("C:/Users/brydu/OneDrive/Desktop/Comps/Image-Segmentation-Senior-Seminar/threshold_segments/segmentation" + file)

if __name__ == "__main__":
    img = import_berkeley("C:\\Users\\brydu\\OneDrive\\Desktop\\Comps\\Image-Segmentation-Senior-Seminar\\threshold_segments_truths\\293029.seg")
    img = Image.fromarray(img * 50)
    img.show()
    '''
    images = []
    color_thresholding_images = otsu_color_thresholding(img)
    for segments in color_thresholding_images:
        new_image = Image.fromarray(segments * 200)
        new_image.show()
        images.append(new_image)
    new_image = Image.merge("RGB", tuple(images))
    new_image.show();'''
    #segment_all_images()


'''def otsu_color_thresholding(image):

    red = get_color_thresholds(image, "R")
    blue = get_color_thresholds(image, "G")
    green = get_color_thresholds(image, "B")
    return (create_image(red, image, "R"), create_image(blue, image, "G"),create_image(green, image, "B"))

def get_color_thresholds(image, color):
    a = []
    if color == "R":
        a = np.array(image.histogram()[0:256])
    elif color == "G":
        a = np.array(image.histogram()[256:512])
    else:
        a = np.array(image.histogram()[512:768])
    return np.where(a == threshold_otsu(a))
    '''
