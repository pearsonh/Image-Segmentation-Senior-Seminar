from PIL import Image
import random
import math
import numpy as np
from operator import add
from operator import sub
import time
from scipy.spatial import distance

#kmeans extensions:
# --random seeds, run multiple times and average
# --chosen seeds somehow: pick first at random, use prob function to place laters far away
# --use physical distance as well as rgb??? normalize, play with weights?
# --probabilistic (expectation maximization)
# --histograms to determine k
# --elbow method to determine k: run for 1<k<10, plot within-cluster sum of squares, look for "elbow"


#distance metrics:
#__euclidean: standard root of summed squares
#--manhattan: sum of abs differences
#__mahalanobis: more flexible, less circular? euclidean divided by squared variance
#__alternative color measure: HSV

def kmeans(image, k):
    '''Runs kmeans segmentation on the given image into k segments
    '''
    img=np.array(image).astype("int")
    #covariance for mahalanobis
    #reshaped=np.reshape(img, (img.shape[0]*img.shape[1],3))
    #covariance=np.cov(reshaped, rowvar=False)
    #print(covariance)
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

def euclidean(centroid, pixel):
    x=(centroid[0]-pixel[0])**2
    y=(centroid[1]-pixel[1])**2
    z=(centroid[2]-pixel[2])**2
    #technically distance is the sqrt of x+y+z, but for comparison it doesn't matter
    #dist=math.sqrt(x+y+z)
    return x+y+z

def find_closest_centroid(pixel, centroids, centroid_totals, centroid_total_counts):
    '''Finds which cluster each pixel would currently belong to,
    and updates the running totals of pixel values and counts per cluster'''
    distances = []
    for value in centroids:
        #possibly mahalanobis is a better distance calculation according to research, but doesn't actually seem better and is very slow
        #distances.append(distance.mahalanobis(value, pixel, covariance))
        distances.append(euclidean(value, pixel))
    ind = np.argmin(distances)
    centroid_totals[ind]+=pixel
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
    #HSV is potentially better than rgb, needs more testing
    img=img.convert("HSV")
    img = Image.fromarray(kmeans(img, 4)*200)
    stop=time.time()
    print("runtime is", stop-start)
    img.show()
