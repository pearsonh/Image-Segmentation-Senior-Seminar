import numpy as np
from PIL import Image
from scipy import *
from scipy.sparse import *
from scipy.sparse.linalg import *
import time
from eval import bde, region_based_eval
from parser import import_berkeley
from tqdm import tqdm

'''Compute difference between two pixels in the np array of rgb tuples by euclidean
    distance between rgb values and euclidean distance between pixel locations'''
def differenceMetricRGBAndDist(coords1, coords2, pixels):
    rgb1 = pixels[coords1[0], coords1[1]]
    rgb2 = pixels[coords2[0], coords2[1]]
    featureDifference = (rgb1[0] - rgb2[0])**2 + (rgb1[1] - rgb2[1])**2 + (rgb1[2] - rgb2[2])**2
    spatialDistance = (coords1[0] - coords2[0])**2 + (coords1[1] - coords2[1])**2
    # feature weight and spatial weight correspond to the sigma values in the edge weight equation, can be tweaked
    featureWeight = 1
    spatialWeight = 1

    # this version of the distance metric equation is what is specified in the Shi and Malik paper,
    # but results in images that just look like static
    # difference = math.exp(-(featureWeight * featureDifference + spatialWeight* spatialDistance))

    # this version of the distance metric is wrong because it assigns higher weights
    # to less similar pixels rather than lower weights. but this is the version that
    # was used to run on all our images before the bug was caught.
    difference = featureWeight * featureDifference + spatialWeight* spatialDistance
    return difference

'''Compute difference between two pixels in the np array of rgb tuples by euclidean
    distance between rgb values'''
def differenceMetricRGB(coords1, coords2, pixels):
    rgb1 = pixels[coords1[0], coords1[1]]
    rgb2 = pixels[coords2[0], coords2[1]]
    # this version of the distance metric is wrong because it assigns higher weights
    # to less similar pixels rather than lower weights. but this is the version that
    # was used to run on all our images before the bug was caught.
    return ((rgb1[0] - rgb2[0])**2 + (rgb1[1] - rgb2[1])**2 + (rgb1[2] - rgb2[2])**2)

'''D is the diagonal matrix and W is the adjacency matrix of edge weights (both csc sparse matrices)
    uses scipy sparse linalg eigenvector solver for the Lanczos method
    returns a tuple containing an array of eigenvalues and an array of eigenvectors'''
def findEigens(D, W):
    eigs = eigsh(inv(D)*(D - W))
    return eigs

'''Takes in an np array of pixels and returns a sparse adjacency matrix (csc_matrix) with edge weights for this image'''
def pixelsToAdjMatrix(pixels, edge_weight_function, connectivity):
    r = connectivity # 12 was the value used in running our experiments
    y,x,_ = pixels.shape #assuming tuples of 3 rgb values are the third coordinate of the shape
    N = x * y
    row = []
    col = []
    data = []
    #go through each pixel in the image and compare it to the r pixels on all sides of it
    for i in tqdm(range(x)):
        for j in range(y):
            # compare (i,j) to each pixel in range r arround it
            for k in range(i-r,i+r): # x coordinate of offset pixel
                for l in range(j-r,j+r): # y coordinate of offset pixel
                    if k >= 0 and l >= 0 and k < x and l < y: # make sure this pixel isn't out of bounds
                        diff = edge_weight_function((j,i), (l,k), pixels)
                        row.append(j*x + i) #add x coord to list of x coords
                        col.append(l*x + k) #add y coord to list of y coords
                        data.append(diff) #add the value that belongs at (j*x + i, l*x + k) in the adjacency matrix
    return csc_matrix((np.array(data), (np.array(row), np.array(col))), shape=(N, N))

'''Takes in a csc_matrix and returns a diagonal matrix (scipy.sparse.dia.dia_matrix) converted to a csc_matrix'''
def adjMatrixToDiagMatrix(matrix):
    N, _ = matrix.shape
    vec = np.array(matrix.sum(axis=0))[0]
    # change any 0s on the diagonal to 1s (otherwise you can get a singular matrix)
    # didn't have time to decide if this was a good way to handle this case
    vec = np.where(vec != 0, vec, 1)
    return diags(vec, offsets=0).tocsc()

'''Takes in an image to segment by mincut, returns an np array of the segment numbers
    for each pixel'''
def mincut(img, edge_weight_function, connectivity=12):
    pixels = np.array(img).astype("int")
    y,x,_ = pixels.shape
    W = pixelsToAdjMatrix(pixels, edge_weight_function, connectivity)
    D = adjMatrixToDiagMatrix(W)
    eigenStuff = findEigens(D, W)
    # pick second smallest eigenvector (second column of the array of eigenvectors) - returns as an ndarray
    eigVec = eigenStuff[1][:, 1]
    # converts eigvec to an indicator vector with 1 if the value is > 0 and 0 otherwise
    eigVecIndicator = (eigVec > 0).astype(int)
    # reshape the indicator vector into a matrix the same shape as pixels
    newEigIndicator = np.reshape(eigVecIndicator, (y,x))
    newEigIndicator = (newEigIndicator).astype('uint8')
    return newEigIndicator

if __name__ == "__main__":
    #sample run of mincut on a berkeley image
    filename = "15088.jpg"
    img = Image.open(filename)
    print(filename)
    start=time.time()
    array = mincut(img, differenceMetricRGB)
    stop=time.time()
    print("total runtime is", stop-start)
    img = Image.fromarray(array*255, mode="L")
    img.show()
    img.save("15088-12-fweight.jpg", "JPEG")
    groundTruth = import_berkeley("15088.seg")
    print("region based is ", region_based_eval(groundTruth, array))
    print("edge based is ", bde(groundTruth, array))
