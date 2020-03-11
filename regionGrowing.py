import PIL
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import sys
import math
from scipy.stats import norm
from skimage.filters import threshold_otsu
import random
import scipy.ndimage as sp
import scipy.optimize as opt
visited = []


def ssd(counts, centers):
    '''Taken from https://bic-berkeley.github.io/psych-214-fall-2016/otsu_threshold.html. Is used in Otsu's Method calculations. Finds the sum of squared deviations from mean '''

    n = np.sum(counts)
    mu = np.sum(centers * counts) / n
    return np.sum(counts * ((centers - mu) ** 2))

def getThresh(img, seed):
    '''Uses Otsu's Method to calculate the optimal threshold. Takes as input an image and a seed value. Outputs a threshold value for the image. The Otsu's Method section of this method is taken from https://bic-berkeley.github.io/psych-214-fall-2016/otsu_threshold.html. Otsu's Method is a well-known algorithm and is a tool used widely to capture the threshold value.'''

    pixList = []
    disList = []
    x=np.array(img).astype("int")
    for item in x:
        for pixel in item:
            pixList.append(pixel)

    i = 0
    while i < len(pixList)-1:
        first = pixList[i]
        R1 = first[0]
        G1 = first[1]
        B1 = first[2]
        second = x[seed[0],seed[1]]
        R2 = second[0]
        G2 = second[1]
        B2 = second[2]
        # calculates euclidean distance between two pixels
        distance = math.sqrt((R2-R1)**2 + (G2-G1)**2 + (B2-B1)**2)
        disList.append(distance)
        i = i+1
    # produces a histogram for the image color intensity values
    n_bins = 10
    x = disList
    counts, edges = np.histogram(x, bins= n_bins)
    bin_centers = edges[:-1] + np.diff(edges) / 2.

    total_ssds = []
    for bin_no in range(1, n_bins):
        left_ssd = ssd(counts[:bin_no], bin_centers[:bin_no])
        right_ssd = ssd(counts[bin_no:], bin_centers[bin_no:])
        total_ssds.append(left_ssd + right_ssd)
    z = np.argmin(total_ssds)
    t = bin_centers[z]
    print('Otsu threshold: ', bin_centers[z])

    npa = np.asarray(x, dtype=np.float32)
    threshold_otsu(npa, n_bins)
    np.allclose(threshold_otsu(npa, n_bins), t)
    return bin_centers[z]

def region_growing(img, seed, threshold):
    '''Performs the actual image segmentation. Takes as input an image, a seed, and a threshold. Outputs a segmentation of the given image as an array with black and white values.'''

    x=np.array(img).astype("int")
    neighbors = [(-1, -1), (0, -1), (1, -1), (-1, 0),  (1, 0), (-1, 1), (0, 1), (1,1)]
    region_size = 1
    distance = 0
    neighbor_points_list = []
    neighbor_color_list = []
    region_mean = x[seed[0],seed[1]]

    a = x.shape
    height = a[0]
    width = a[1]
    image_size = height * width

    # produces an empty white array
    totalArray = []
    segmented_img = np.zeros((height, width, 3), np.uint8)
    # set the region seed to white
    segmented_img[seed[0], seed[1]] = [255, 255, 255]

    visited.append(seed)
    newMin = threshold
    while(region_size < image_size):
        # iterate through the 8-connected neighbors
        for i in range(8):
            x_new = seed[0] + neighbors[i][0]
            y_new = seed[1] + neighbors[i][1]

            check_inside = (x_new >= 0) & (y_new >= 0) & (x_new < height) & (y_new < width)
            # check edge cases
            if check_inside:
                if [x_new, y_new] not in visited and [x_new, y_new] not in neighbor_points_list:
                    newArr = []
                    neighbor_points_list.append([x_new, y_new])
                    one = x[x_new, y_new][0]
                    two = x[x_new, y_new][1]
                    three = x[x_new, y_new][2]
                    newArr.append(one)
                    newArr.append(two)
                    newArr.append(three)
                    totalArray.append(newArr)
                    neighbor_color_list = totalArray

        values = []
        R1 = region_mean[0]
        G1 = region_mean[1]
        B1 = region_mean[2]
        for item in neighbor_color_list:
            R2 = item[0]
            G2 = item[1]
            B2 = item[2]
            # compute euclidean distance between each neighbor and the region mean value
            distance = math.sqrt((R2-R1)**2 + (G2-G1)**2 + (B2-B1)**2)
            values.append(distance)
        # select the minimum distance value (most similar pixel to region mean)
        newMin = min(values)
        ind = values.index(newMin)
        index = neighbor_points_list[ind]
        otherindex = neighbor_color_list[ind]

        # compare minimum distance value to threshold
        if newMin < threshold:
            visited.append(index)
            # set the pixel to white and add to region
            segmented_img[index[0], index[1]] = [255, 255, 255]
            region_size += 1
            neighbor_points_list.remove([index[0], index[1]])
            newLst = []
            ans = x[index[0], index[1]]
            first = ans[0]
            second = ans[1]
            third = ans[2]
            newLst.append(first)
            newLst.append(second)
            newLst.append(third)
            neighbor_color_list.remove(newLst)
            seed = index
        else:
            break

    seg = np.swapaxes(segmented_img,0,1)
    seg = np.swapaxes(seg,1,0)
    new_im = Image.fromarray(seg)
    new_im.show()
    # return seg

    a = np.copy(new_im)
    a = a / 255
    a = a[:,:,0]
    return a

if __name__ == '__main__':
    '''Main function to run region Growing algorithm on a file, output a segmentation, and calculate different evaluation metrics'''

    file = "22093.jpg"
    img = Image.open(file)
    # randomly selects seed
    int1 = random.randrange(0, 255)
    int2 = random.randrange(0, 255)
    seed = [int1, int2]
    print('Seed:', seed)
    # get threshold
    threshold = getThresh(img, seed)
    # resize image for shorter runtime
    newsize = (255,255)
    img = img.resize(newsize)
    # perform segmentation and show segmented image
    output = region_growing(img, seed, threshold)
    # seg = np.swapaxes(seg,0,1)
    # seg = np.swapaxes(seg,1,0)
    # new_im = Image.fromarray(seg)

    # array of 1s and 0s for use in metrics
