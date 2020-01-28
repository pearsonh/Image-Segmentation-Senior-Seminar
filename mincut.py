import numpy as np
from PIL import Image
from scipy import *
from scipy.sparse import *
from scipy.sparse.linalg import *

'''def pixelValsToAdjMatrix2(pixels):
    y,x,_ = pixels.shape
    N = x * y
    matrix = np.zeros(shape=(N, N))
    for i in range(x):
        for j in range(y):
            for k in range(x):
                for l in range(y):
                    matrix[l*x + k, j*x + i] = differenceMetric((j,i), (l,k), pixels)
    return matrix

def pixelValsToAdjMatrix(pixels):
    y,x,_ = pixels.shape
    N = x * y
    matrix = lil_matrix((N, N), dtype='d') #scipy sparse function
    for i in range(x):
        for j in range(y):
            for k in range(x):
                for l in range(y):
                    diff = differenceMetric((j,i), (l,k), pixels)
                    if (diff < 20000): # some threshold, probably use distance away in reality
                        matrix[l*x + k, j*x + i] = diff
    return matrix'''

# compute difference between two pixels in the np array of rgb tuples by euclidean
#distance (or replace with some other metric in this method)
def differenceMetric(coords1, coords2, pixels):
    rgb1 = pixels[coords1[0], coords1[1]]
    rgb2 = pixels[coords2[0], coords2[1]]
    return ((rgb1[0] - rgb2[0])**2 + (rgb1[1] - rgb2[1])**2 + (rgb1[2] - rgb2[2])**2)

'''def adjMatrixToDiagMatrix2(matrix):
    N,_ = matrix.shape
    diag = np.zeros(shape=(N,N))
    for i in range(N):
        for j in range(N):
            diag[i,i] += matrix[j,i]
    return diag

def adjMatrixToDVector(matrix):
    N,_ = matrix.shape
    vector = np.zeros(shape=(N, 1))
    for i in range(N):
        for j in range(N):
            vector[i] += matrix[j,i]
    #just for testing purposes, to guarantee a non-singular matrix (need to figure out what they do about this in the alg)
    for i in range(N):
        if vector[i] == 0:
            vector[i] = 1
    return vector

def adjMatrixToDiagMatrix(matrix):
    N,_ = matrix.shape
    diag = lil_matrix((N, N), dtype='d') #scipy sparse function
    for i in range(N):
        for j in range(N):
            diag[i,i] += matrix[j,i]
    #just for testing purposes, to guarantee a non-singular matrix (need to figure out what they do about this in the alg)
    for i in range(N):
        if diag[i,i] == 0:
            diag[i,i] = 1
    return diag'''

#D is the diagonal matrix and W is the adjacency matrix of edge weights
#returns ?
def findEigens(D, W):
    #print("Inverse of D", inv(D)) #scipy sparse linalg inverse function
    #print("D - W:", D - W)
    return eigsh(inv(D)*(D - W)) #scipy sparse linalg eigenvector solver for Lanczos method (that they used in the paper)

'''def findEigensDense(D, W):
    print("Inverse of D", np.linalg.inv(D)) #scipy sparse linalg inverse function
    print("D - W:", D - W)
    return np.linalg.eig(np.linalg.inv(D)*(D - W))'''

#takes in an np array of pixels and returns a sparse adjacency matrix (csc_matrix) with edge weights for this image
def pixelsToAdjMatrix(pixels):
    r = 10
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
    return diags(vec, offsets=0).tocsc()

# simple test, doesn't want to find eigens bc the matrix isn't big enough
#pixels = np.array([[(0,0,0),(255,255,255)],[(3,3,3), (1,1,1)]])
# test with actual image
img = Image.open("22093.jpg")
pixels = np.array(img).astype("int")

print("Image:\n", pixels)
W = pixelsToAdjMatrix(pixels)
print("adjacency matrix:\n", W)
D = adjMatrixToDiagMatrix(W)
print("diagonal matrix:\n", D)
print(findEigens(D, W))


#np turns the tuples into nested arrays, which adds another dimension to the shape.
#right now i'm just collecting and ignoring it, but is there any way to keep them as tuples
#pixels = np.array([[(0,0,0),(255,255,255)],[(3,3,3), (1,1,1)]])
'''img = Image.open("22093.jpg")
pixels = np.array(img).astype("int")
print("pixel values:\n", pixels)
matrix = pixelValsToAdjMatrix(pixels)
print("Adjacency matrix:\n", matrix)
#print("dense matrix:\n", np.array(matrix.todense()))
diag = adjMatrixToDiagMatrix(matrix)
print("Diag sparse matrix:", diag)
#print("Diag dense matrix:", diag.todense())
print(findEigens(diag, matrix))'''
