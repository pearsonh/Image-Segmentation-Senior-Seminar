import PIL
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import sys
import math


def region_growing(img, seed):
    x=np.array(img).astype("int")
    neighbors = [(-1, -1), (0, -1), (1, -1), (-1, 0),  (1, 0), (-1, 1), (0, 1), (1,1)]
    region_threshold = 200
    region_size = 1
    distance = 0
    neighbor_points_list = []
    neighbor_color_list = []
    region_mean = x[seed[0],seed[1]]
    a = x.shape
    height = a[0]
    width = a[1]
    image_size = height * width
    visited = []
    totalArray = []
    segmented_img = np.zeros((height, width, 3), np.uint8)
    segmented_img[seed[0], seed[1]] = [255, 255, 255]

    visited.append(seed)
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
                    # print(x[x_new, y_new])
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
            distance = math.sqrt((R2-R1)**2 + (G2-G1)**2 + (B2-B1)**2)
            values.append(distance)
        newMin = min(values)
        ind = values.index(newMin)
        index = neighbor_points_list[ind]
        otherindex = neighbor_color_list[ind]

        if newMin < region_threshold:
            visited.append(index)
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

    return segmented_img

if __name__ == '__main__':
    img = Image.open("22093.jpg")

    seed = [1, 1]
    newsize = (255,255)
    img = img.resize(newsize)
    seg = region_growing(img, seed)
    seg = np.swapaxes(seg,0,1)
    new_im = Image.fromarray(seg)
    new_im.show()
