import numpy as np
from PIL import Image
from scipy import *
from scipy.sparse import *
from scipy.sparse.linalg import *
np.set_printoptions(threshold=np.inf)

# compute difference between two pixels in the np array of rgb tuples by euclidean
#distance (or replace with some other metric in this method)
def differenceMetric(coords1, coords2, pixels):
    rgb1 = pixels[coords1[0], coords1[1]]
    rgb2 = pixels[coords2[0], coords2[1]]
    return ((rgb1[0] - rgb2[0])**2 + (rgb1[1] - rgb2[1])**2 + (rgb1[2] - rgb2[2])**2)

#D is the diagonal matrix and W is the adjacency matrix of edge weights
#returns a tuple with an array of eigenvalues and an array of eigenvectors
def findEigens(D, W):
    #print("Inverse of D", inv(D)) #scipy sparse linalg inverse function
    #print("D - W:", D - W)
    return eigsh(inv(D)*(D - W)) #scipy sparse linalg eigenvector solver for Lanczos method (that they used in the paper)

#takes in an np array of pixels and returns a sparse adjacency matrix (csc_matrix) with edge weights for this image
def pixelsToAdjMatrix(pixels):
    r = 18
    y,x,_ = pixels.shape #assuming tuples of 3 values are the third coordinate of the shape
    N = x * y
    row = []
    col = []
    data = []
    #go through each pixel in the image and compare it to the r pixels on all sides of it
    for i in range(x):
        for j in range(y):
            # compare (i,j) to each pixel in range r arround it
            for k in range(i-r,i+r): # x coordinate of offset pixel
                for l in range(j-r,j+r): # y coordinate of offset pixel
                    if k >= 0 and l >= 0 and k < x and l < y: # make sure this pixel isn't out of bounds
                        diff = differenceMetric((j,i), (l,k), pixels)
                        #if (diff < 20000): # some threshold, probably dont need this once im using a distance threshold
                            #matrix[l*x + k, j*x + i] = diff
                        row.append(j*x + i) #add x coord to list of x coords
                        col.append(l*x + k) #add y coord to list of y coords
                        data.append(diff) #add the value that belongs at (j*x + i, l*x + k) in the adjacency matrix
    return csc_matrix((np.array(data), (np.array(row), np.array(col))), shape=(N, N))

# takes in a csc_matrix and returns a diagonal matrix (scipy.sparse.dia.dia_matrix) converted to a csc_matrix
def adjMatrixToDiagMatrix(matrix):
    N, _ = matrix.shape
    #sum outputs a numpy matrix of the same shape of matrix (so it's an array
    #with an array inside), and I can't' figure out how to access just the first row of it
    #other than by changing it to an np array first
    vec = np.array(matrix.sum(axis=0))[0]
    vec = np.where(vec != 0, vec, 1)
    return diags(vec, offsets=0).tocsc()

# simple test, doesn't want to find eigens bc the matrix isn't big enough
#pixels = np.array([[(0,0,0),(255,255,255)],[(3,3,3), (1,1,1)]])
# test with actual image
img = Image.open("test4.jpg")
pixels = np.array(img).astype("int")

#print("Image:\n", pixels)
W = pixelsToAdjMatrix(pixels)
#print("adjacency matrix:\n", W)
D = adjMatrixToDiagMatrix(W)
#print("diagonal matrix:\n", D)
eigenStuff = findEigens(D, W)
#print(eigenStuff)
eigVec = eigenStuff[1][:, 1] # pick second smallest eigenvector (second column of the array of eigenvectors) - returns as an ndarray
#print(eigVec)
eigVecIndicator = (eigVec > 0).astype(int) # converts eigvec to an indicator vector with 1 if the value is > 0 and 0 otherwise
#print(eigVecIndicator)

grayImage = img.convert(mode="L")
grayPixels = np.array(grayImage).astype("int")
#print(grayPixels)
y,x = grayPixels.shape
y,x,_ = pixels.shape
newEigIndicator = np.reshape(eigVecIndicator, (y,x)) # reshape the indicator vector into a matrix the same shape as graypixels
# print(newEigIndicator[:, :36])
# print(newEigIndicator[:, 36:])

newPixels = (grayPixels * newEigIndicator).astype('uint8')
# for some reason the image only displays correctly if you make sure the type is uint8
newEigIndicator = (newEigIndicator * 255).astype('uint8')
#print(newEigIndicator)
# print(newEigIndicator[:, :36])
# print(newEigIndicator[:, 36:])
#print(newPixels)
img = Image.fromarray(newEigIndicator, mode="L")
img.show()
img.save("test4-18-1s.jpg", "JPEG")
#
# img = Image.fromarray(newPixels, mode="L")
# img.show()
