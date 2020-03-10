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

    return returnImage

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


if __name__ == "__main__":
    img = import_berkeley("C:\\Users\\brydu\\OneDrive\\Desktop\\Comps\\Image-Segmentation-Senior-Seminar\\threshold_segments_truths\\293029.seg")
    img = Image.fromarray(img * 50)
    img.show()

