from PIL import Image
import random
import math
import numpy as np
from operator import add
from operator import sub
import time

def kmeans(image, k):
    '''Runs kmeans segmentation on the given image into k segments
    '''
    img=np.array(image).astype("int")
    centroids = []
    centroid_means = np.zeros((k,3))
    #assigns k random starting centroids
    for i in range(k):
        new = [0,0]
        random.seed(0)
        new[0] = random.randrange(image.width)
        new[1] = random.randrange(image.height)
        #checks to avoid duplicate centroids
        while new in centroids:
            new[0] = random.randrange(image.width)
            new[1] = random.randrange(image.height)
        centroids.append(new)
        centroid_means[i]=img[new[1],new[0]]
    old_means=np.zeros((len(centroid_means),3))
    #continously updates the centroids and cluster assignments
    while True:
        image_matrix, centroid_means = cluster(img, centroid_means)
        #accepts the image matrix if no centroid value has changed by more than 1.0
        if np.allclose(centroid_means, old_means, atol=1.0, rtol=0.0):
            return image_matrix
        old_means=centroid_means
    return image_matrix

def find_closest_centroid(rgb, centroids, centroid_totals, centroid_total_counts):
    '''Finds which cluster each pixel would currently belong to,
    and updates the running totals of pixel values and counts per cluster'''
    distances = []
    for value in centroids:
        x=(value[0]-rgb[0])**2
        y=(value[1]-rgb[1])**2
        z=(value[2]-rgb[2])**2
        #technically distance is the sqrt of x+y+z, but for comparison it doesn't matter
        #dist=math.sqrt(x+y+z)
        distances.append(x+y+z)
    ind = np.argmin(distances)
    centroid_totals[ind]+=rgb
    centroid_total_counts[ind]+=1
    return float(ind)


def cluster(image, centroids):
    '''Takes in a matrix of pixels and one of cluster centroids, reassigns
    pixels to their new closest clusters, and updates cluster centroids.
    Returns a matrix of pixel cluster assingments and the updated centroid values
    '''
    centroid_totals = np.zeros((len(centroids),3))
    centroid_total_counts = np.zeros(len(centroids))
    image_matrix = np.zeros((len(image), len(image[0])))
    #currently it is quicker to loop and call the function rather than use numpy
    #image_matrix=np.apply_along_axis(find_closest_centroid, -1, image, centroids, centroid_totals, centroid_total_counts)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_matrix[i][j] = find_closest_centroid(image[i][j], centroids, centroid_totals, centroid_total_counts)
    cluster_means=[]
    #currently it is quicker to loop rather than use numpy indexing
    #cluster_means=centroid_totals/centroid_total_counts[:, None]
    #computes average rgb values for each cluster
    for i in range(len(centroid_totals)):
        total=centroid_totals[i]
        sum=centroid_total_counts[i]
        cluster_means.append([total[0]/sum, total[1]/sum, total[2]/sum])
    return image_matrix, cluster_means

if __name__ == "__main__":
    start=time.time()
    img = Image.open("22093.jpg")
    img = Image.fromarray(kmeans(img, 4)*200)
    stop=time.time()
    print("runtime is", stop-start)
    img.show()
