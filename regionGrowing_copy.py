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

def getSeed():
    '''Calculates a random seed value in the image to start the segmentation from. Outputs a 2-tuple with format [row, col] to access an individual pixel.'''

    int1 = random.randrange(0, 255)
    int2 = random.randrange(0, 255)
    seed = (int1, int2)
    if seed in visited:
        getSeed()
    else:
        return seed

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

    return segmented_img

def getSegments(seg):
    '''Takes as input an already segmented image (an array with white or black values). Outputs an array of 0s and 1s for the segmented image, which is used to evaluate the segmentation against the ground truth.'''

    a = np.copy(seg)
    a = a / 255
    a = a[:,:,0]
    return a

def import_weizmann(filename):
    '''Given the path to a file containing a Weizmann encoding of a segmentation,
    returns the numpy version of that segmentation'''

    image = Image.open(filename)
    image_matrix=np.array(image)
    segmentedmin = np.argmin(image_matrix, axis=2)
    segmentedmax = np.argmax(image_matrix, axis=2)
    segmented=np.add(segmentedmin, segmentedmax)/2
    return segmented

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

def display_segmentation(image,number_of_segmentations):
    '''given a numpy array with a segmentation, displays that segmentation'''

    img = Image.fromarray(image*255/number_of_segmentations)
    img.show()

def processSegmentationWithKernels(segmentation):
    '''Create a list of pixels that are on the edge of a segment for a given
        segmentation. The input segmentation is a 2D numpy array where each
        element is the label of the segment that pixel falls in, and the output
        is a nested np array of coordinates describing the (x, y) locations of edge pixels.
        Does edge pixel checks by processing the segmentation with kernel convolution.
        Each kernel checks whether one of the four neighboring pixels is in the same
        segment as the middle pixel--if the middle pixel is the same, then in the
        convolved array it will now be 0. By taking the absolute value of each of the
        convolved arrays and taking their sum, any pixel that is non-zero will be an
        edge pixel.
        A pixel is considered 'on an edge' if at least one pixel
        on the four sides of it lies in a different segment (or if it is on the
        actual edge of the image).
        For example, suppose segmentation is the following:
        0 0 0 1 1 1
        0 0 0 1 1 1
        0 0 0 1 1 1
        0 0 0 1 1 1
        Then edge pixels are marked by X's below:
        X X X X X X
        X 0 X X 1 X
        X 0 X X 1 X
        X X X X X X
        Returns a list of all the non-zero elements in this sum array (i.e. all the
        edge pixel coordinates).'''

    kernelTop = np.asarray([[0,-1,0],[0,1,0],[0,0,0]])
    kernelBottom = np.asarray([[0,0,0],[0,1,0],[0,-1,0]])
    kernelLeft = np.asarray([[0,0,0],[-1,1,0],[0,0,0]])
    kernelRight = np.asarray([[0,0,0],[0,1,-1],[0,0,0]])
    convolvedTop = abs(sp.convolve(segmentation, kernelTop))
    convolvedBottom = abs(sp.convolve(segmentation, kernelBottom))
    convolvedLeft = abs(sp.convolve(segmentation, kernelLeft))
    convolvedRight = abs(sp.convolve(segmentation, kernelRight))
    sumArr = convolvedTop + convolvedBottom + convolvedLeft + convolvedRight
    nz = np.nonzero(sumArr)

    #add the coordinates of outer edge pixels of the image
    height,length = segmentation.shape
    top_range = np.expand_dims(np.arange(length),axis=0)
    top_zeros = np.expand_dims(np.zeros(length),axis=0)
    top_edges = np.concatenate((top_zeros,top_range), axis=0)

    bottom_range = np.expand_dims(np.arange(length),axis=0)
    bottom_fill = np.expand_dims(np.full(length, height - 1),axis=0)
    bottom_edges = np.concatenate((bottom_fill,bottom_range), axis=0)

    left_range = np.expand_dims(np.arange(1,height - 1),axis=0)
    left_zeros = np.expand_dims(np.zeros(height - 2),axis=0)
    left_edges = np.concatenate((left_range,left_zeros), axis=0)

    right_range = np.expand_dims(np.arange(1,height -1),axis=0)
    right_fill = np.expand_dims(np.full(height - 2, length - 1),axis=0)
    right_edges = np.concatenate((right_range,right_fill), axis=0)

    nz = np.concatenate((top_edges, bottom_edges, nz, left_edges, right_edges), axis=1)

    # make sure there are no duplicate edges in the list
    coords = np.unique(np.transpose(nz), axis=0)
    return coords

def bde(segmentation,groundtruth):
    ''' Given a segmentation generated by one of our algorithms and a ground truth
        segmentation (both 2D numpy arrays of segment labels corresponding to each pixel),
        return the BDE measure for this pair of segmentations (a positive float).
        The BDE metric is calculated as the average of [the sum, over all the edge pixels
        in segmentation, of the distances from the nearest edge pixel in the ground
        truth] and [the sum, over all the edge pixels in ground truth, of the distances
        from the nearest edge pixel in segmentation]. More broadly, it is a measure
        of how closely the edges in segmentation match the edges in groundtruth.
        A small BDE measure means a closer fit for edges.'''

    height,length = segmentation.shape
    # locate edge pixels in each segmentation and store these as lists of coordinates
    segEdges = processSegmentationWithKernels(segmentation)
    truEdges = processSegmentationWithKernels(groundtruth)
    numSegEdges, _ = segEdges.shape
    numTruEdges, _ = truEdges.shape
    # create matrices of tuples whose rows/cols are copies of segEdges and truEdges
    # (we get A and B with dimensions numSegEdges by numTruEdges, with the tuples
    # corresponding to true edges on the rows of A and the tuples corresponding
    # to seg edges on the columns of B)
    A = np.tile(truEdges, (numSegEdges, 1))
    B = np.tile(segEdges, (numTruEdges, 1))
    A = np.reshape(A, (numSegEdges, numTruEdges, 2))
    B = np.transpose(np.reshape(B, (numTruEdges, numSegEdges, 2)), (1,0,2))
    # C is a matrix of euclidean distances between every pair of true and segEdges
    C = np.sqrt(np.sum(np.square(A - B), axis=2))
    # find the min value for each row and each column of C.
    # this is equivalent to finding the min distance from each true edge to a seg
    # edge, and the min distance from each seg edge to a true edge
    minsTruToSeg = np.sum(np.amin(C, axis=0))
    minsSegToTru = np.sum(np.amin(C, axis=1))
    return (minsTruToSeg / len(truEdges) + minsSegToTru / len(segEdges)) / 2

def region_based_eval(truth, generated):
    ''' Calculates a Jaccard score for the similarity of the generated segmentation to the
    ground truth segmentation, based on a bipartite matching of segments, or regions.
    Given the ground truth and generated segmentation as 2D numpy arrays of
    segment labels for each pixel, decide which segments best correspond in ground
    truth and generated segmentation, then return the Jaccard measure (between 0 and 1)
    for how closely these segments match up.
    The Jaccard metric is calculated as the average, over all ground truth/generated
    segmentation region pairs, of the intersection of the corresponding regions
    divided by the union of these regions. A larger Jaccard measure means
    a closer overlap in regions.'''

    height,length = truth.shape
    height2,length2 = generated.shape
    height=max(height, height2)
    length=max(length, length2)
    generated=generated.astype(int)
    truth=truth.astype(int)
    max_true=np.amax(truth)+1
    max_alg=np.amax(generated)+1 #axis??
    weights =  np.array([[height*length]*max_alg]*max_true)
    true_sizes=np.zeros(max_true)
    alg_sizes=np.zeros(max_alg)
    for i in range(height):
        for j in range(length):
            weights[truth[i,j], generated[i,j]]-=1
            true_sizes[truth[i,j]]+=1
            alg_sizes[generated[i,j]]+=1

    true_ind, alg_ind=opt.linear_sum_assignment(weights)
    total=0
    count=0
    for i in range(len(alg_ind)):
        if alg_sizes[alg_ind[i]]!=0:
            count+=1
            match = weights[true_ind[i], alg_ind[i]]
            intersect = height*length-match
            jaccard = intersect/(alg_sizes[alg_ind[i]]+true_sizes[true_ind[i]]-intersect)
            total+=jaccard
    return total/count

if __name__ == '__main__':
    '''Main function to run region Growing algorithm on a file, output a segmentation, and calculate different evaluation metrics'''

    file = "stp.jpg"
    # img = Image.open("final_images_berkeley/" + file)
    img = Image.open(file)
    int1 = random.randrange(0, 255)
    int2 = random.randrange(0, 255)
    seed = [int1, int2]
    print(seed)
    newsize = (255,255)
    threshold = getThresh(img, seed)
    # img = img.resize(newsize)
    seg = region_growing(img, seed, threshold)
    seg = np.swapaxes(seg,0,1)
    seg = np.swapaxes(seg,1,0)

    new_im = Image.fromarray(seg)
    new_im.show()

    # truth = import_berkeley("berkeley_truth/3096.seg")
    # display_segmentation(truth,12)
    output = getSegments(seg)
    # print(output)
    # accuracy = bde(output, truth)
    # print(accuracy)

    # jaccard_score = region_based_eval(truth, output)
    # print("jaccard score is:", jaccard_score)
