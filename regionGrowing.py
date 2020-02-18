import PIL
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import sys
import math
from datetime import datetime
from scipy.stats import norm
from skimage.filters import threshold_otsu
import random

start=datetime.now()
visited = []

def ssd(counts, centers):
    """ Sum of squared deviations from mean """
    n = np.sum(counts)
    mu = np.sum(centers * counts) / n
    return np.sum(counts * ((centers - mu) ** 2))

def getThresh():
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
        distance = math.sqrt((R2-R1)**2 + (G2-G1)**2 + (B2-B1)**2)
        disList.append(distance)
        i = i+1
    n_bins = 10
    x = disList
    # n, bins, patches = plt.hist(x, n_bins, facecolor='blue', alpha=0.5)
    counts, edges = np.histogram(x, bins= n_bins)
    bin_centers = edges[:-1] + np.diff(edges) / 2.

    total_ssds = []
    for bin_no in range(1, n_bins):
        left_ssd = ssd(counts[:bin_no], bin_centers[:bin_no])
        right_ssd = ssd(counts[bin_no:], bin_centers[bin_no:])
        total_ssds.append(left_ssd + right_ssd)
    z = np.argmin(total_ssds)
    t = bin_centers[z]
    print('Otsu threshold (c[z]):', bin_centers[z])
    # Otsu threshold (c[z]): 0.33984375

    npa = np.asarray(x, dtype=np.float32)
    threshold_otsu(npa, n_bins)
    np.allclose(threshold_otsu(npa, n_bins), t)

    # plt.show()
    return bin_centers[z]

def getSeed():
    int1 = random.randrange(0, 255)
    int2 = random.randrange(0, 255)
    seed = (int1, int2)
    if seed in visited:
        getSeed()
    else:
        return seed

def region_growing(img, seed, threshold):
    x=np.array(img).astype("int")
    neighbors = [(-1, -1), (0, -1), (1, -1), (-1, 0),  (1, 0), (-1, 1), (0, 1), (1,1)]
    region_threshold = threshold
    region_size = 1
    distance = 0
    neighbor_points_list = []
    neighbor_color_list = []
    region_mean = x[seed[0],seed[1]]
    # region_mean = [0,0,0]
    # region_mean[0] += x[seed[0],seed[1]][0]
    # region_mean[1] += x[seed[0],seed[1]][1]
    # region_mean[2] += x[seed[0],seed[1]][2]

    a = x.shape
    height = a[0]
    width = a[1]
    image_size = height * width

    totalArray = []
    segmented_img = np.zeros((height, width, 3), np.uint8)
    segmented_img[seed[0], seed[1]] = [255, 255, 255]

    visited.append(seed)
    # segmented_img[seed[0], seed[1]] = [255, 255, 255]
    newMin = region_threshold
    while(region_size < image_size):
        for i in range(8):
            x_new = seed[0] + neighbors[i][0]
            y_new = seed[1] + neighbors[i][1]

            check_inside = (x_new >= 0) & (y_new >= 0) & (x_new < height) & (y_new < width)

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
            # print("NEW ITEM")
            R2 = item[0]
            G2 = item[1]
            B2 = item[2]
            distance = math.sqrt((R2-R1)**2 + (G2-G1)**2 + (B2-B1)**2)
            values.append(distance)
        newMin = min(values)
        # print(newMin)
        ind = values.index(newMin)
        index = neighbor_points_list[ind]
        # print(index)
        # print(region_mean)
        otherindex = neighbor_color_list[ind]


        if newMin < region_threshold:
            visited.append(index)
            # region_mean[0] = (region_mean[0] + otherindex[0]) / 2
            # region_mean[1] = (region_mean[1] + otherindex[1]) / 2
            # region_mean[2] = (region_mean[2] + otherindex[2]) / 2
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

    #
    # seed = getSeed()
    # segmented_img[seed[0], seed[1]] = [255, 255, 255]
    # visited.append(seed)
    # region_mean = x[seed[0],seed[1]]
    # newMin = region_threshold
    # while(region_size < image_size):
    #     for i in range(8):
    #         x_new = seed[0] + neighbors[i][0]
    #         y_new = seed[1] + neighbors[i][1]
    #
    #         check_inside = (x_new >= 0) & (y_new >= 0) & (x_new < height) & (y_new < width)
    #
    #         if check_inside:
    #             if [x_new, y_new] not in visited and [x_new, y_new] not in neighbor_points_list:
    #                 newArr = []
    #                 neighbor_points_list.append([x_new, y_new])
    #                 one = x[x_new, y_new][0]
    #                 two = x[x_new, y_new][1]
    #                 three = x[x_new, y_new][2]
    #                 newArr.append(one)
    #                 newArr.append(two)
    #                 newArr.append(three)
    #                 totalArray.append(newArr)
    #                 neighbor_color_list = totalArray
    #
    #     values = []
    #     R1 = region_mean[0]
    #     G1 = region_mean[1]
    #     B1 = region_mean[2]
    #     for item in neighbor_color_list:
    #         # print("NEW ITEM")
    #         R2 = item[0]
    #         G2 = item[1]
    #         B2 = item[2]
    #         distance = math.sqrt((R2-R1)**2 + (G2-G1)**2 + (B2-B1)**2)
    #         values.append(distance)
    #     newMin = min(values)
    #     # print(newMin)
    #     ind = values.index(newMin)
    #     index = neighbor_points_list[ind]
    #     # print(index)
    #     # print(region_mean)
    #     otherindex = neighbor_color_list[ind]
    #
    #     if newMin < region_threshold:
    #         visited.append(index)
    #         # region_mean[0] = (region_mean[0] + otherindex[0]) / 2
    #         # region_mean[1] = (region_mean[1] + otherindex[1]) / 2
    #         # region_mean[2] = (region_mean[2] + otherindex[2]) / 2
    #         segmented_img[index[0], index[1]] = [255, 255, 255]
    #         region_size += 1
    #         neighbor_points_list.remove([index[0], index[1]])
    #         newLst = []
    #         ans = x[index[0], index[1]]
    #         first = ans[0]
    #         second = ans[1]
    #         third = ans[2]
    #         newLst.append(first)
    #         newLst.append(second)
    #         newLst.append(third)
    #         neighbor_color_list.remove(newLst)
    #         seed = index
    #
    #     else:
    #         break

    return segmented_img

def getSegments(seg):
    a = np.copy(seg)
    for item in a:
        for pixel in item:
            if pixel[0] == 255:
                pixel = 0
            else:
                pixel = 1
    return a

def runRG():
    img = Image.open("42049.jpg")
    int1 = random.randrange(0, 255)
    int2 = random.randrange(0, 255)
    seed = [int1, int2]
    print("SEED")
    print(seed)
    newsize = (255,255)
    threshold = getThresh()
    img = img.resize(newsize)
    seg = region_growing(img, seed, threshold)
    seg = np.swapaxes(seg,0,1)
    seg = np.swapaxes(seg,1,0)
    # print(seg)
    new_im = Image.fromarray(seg)
    new_im.show()
    print(datetime.now()-start)

    output = getSegments(seg)


if __name__ == '__main__':
    # img = Image.open("42049.jpg")
    # int1 = random.randrange(0, 255)
    # int2 = random.randrange(0, 255)
    # seed = [int1, int2]
    # print("SEED")
    # print(seed)
    # newsize = (255,255)
    # threshold = getThresh()
    # img = img.resize(newsize)
    # seg = region_growing(img, seed, threshold)
    # seg = np.swapaxes(seg,0,1)
    # seg = np.swapaxes(seg,1,0)
    # # print(seg)
    # new_im = Image.fromarray(seg)
    # new_im.show()
    # print(datetime.now()-start)
    #
    # output = getSegments(seg)
