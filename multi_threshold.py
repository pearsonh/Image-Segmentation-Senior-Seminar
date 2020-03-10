from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu, threshold_multiotsu
import os
from collections import Counter


def otsu_greyspace_thresholding(image):
    '''A function that takes in a PIL Image object and outputs a 2d numpy
    array of segmentation assignments per pixel using Otsu's method on
    a grayscale histogram'''


    #Creates a greyscale version of the input image
    imageGrey = image.convert(mode="L")

    x, y = imageGrey.size
    #Gets a threshold for the image
    threshold = get_histogram_threshold(imageGrey)[0]
    returnImage = np.zeros(shape=(y,x))
    segments = len(threshold)
    #Maps 1's and 0's based on whether the pixel values are above or below the threshold.
    if len(threshold) is 1:
        for j in range(x):
            for i in range(y):
                returnImage[i,j] = 1 if imageGrey.getpixel((j,i)) > threshold else 0
    else:
        #This code is no longer used, as it was specified that threshold is now the
        #first element. Previously, if Otsu's found two values with the same minimum
        #variance both would be returned.
        for j in range(x):
            for i in range(y):
                if imageGrey.getpixel((j,i)) < threshold[0]:
                    returnImage[i,j] = 0
                if imageGrey.getpixel((j,i)) > threshold[0]:
                    returnImage[i,j] = 1
                if imageGrey.getpixel((j,i)) > threshold[1]:
                    returnImage[i,j] = 2

    return returnImage

def otsu_multi_greyspace_thresholding(image, segments):
    '''Takes in a PIL image and a number of segments, and runs the
    multi-threshold variant of Otsu's with the specified hyperparameter.
    Outputs a 2d array with pixel segment assignments.'''

    imageGrey = image.convert(mode="L")

    x, y = imageGrey.size
    #Gets a threshold for the image
    thresholds = threshold_multiotsu(np.array(image), segments)

    #While thresholds are supposed to already be sorted properly,
    #this ensures that they are.
    thresholds = sorted(thresholds)
    returnImage = np.zeros(shape=(y,x))
    #Assigns segments to each pixel based off of which threshold it is above.
    #Threshold
    for j in range(x):
        for i in range(y):
            segment = 0
            for p in range(segments - 1):
                if imageGrey.getpixel((j,i)) > thresholds[p]:
                    segment = p + 1
            returnImage[i,j] = segment

    return returnImage

def smoothy(image_array):
    '''Experimental function to smooth a segmentation. Ran into issues implementing,
    but have left the code here.'''
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
    '''Experimental helper function, no longer necessary'''
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


if __name__ == "__main__":
    '''When used as the main function, runs Otsu's grayspace on
    our default image and displays it. Then runs multi Otsu's and
    displays it'''

    temp = Image.open("22093.jpg")
    img = otsu_greyspace_thresholding(temp)
    img = Image.fromarray(img * 255)
    img.show()
    
    img2 = otsu_multi_greyspace_thresholding(temp, 3)
    img2 = Image.fromarray(img2 * 100)
    img2.show()
