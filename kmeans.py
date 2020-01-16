from PIL import Image
import random
import math
import numpy as np
from operator import add
from operator import sub
import time
from scipy.spatial import distance

#seeds:
# --random seeds, run multiple times and average
# --chosen seeds somehow: pick first at random, use prob function to place laters far away

#features:
# __use physical distance as well as rgb??? normalize, play with weights?
# -- texture?

#determining k:
# __histograms to determine k (mean shift?)
# --elbow method to determine k: run for 1<k<10, plot within-cluster sum of squares, look for "elbow"

#misc:
# --probabilistic (expectation maximization)
# --smoothing

#distance metrics:
#__euclidean: standard root of summed squares
#--manhattan: sum of abs differences
#__mahalanobis: more flexible, less circular? euclidean divided by squared variance
#__alternative color measure: HSV

#https://spin.atomicobject.com/2015/05/26/mean-shift-clustering/
#curently prohibitively expensive
def mean_shift(image, bandwidth):
    img=np.array(image).astype("int")
    #reshaped=np.reshape(img, (img.shape[0]*img.shape[1],3))
    #print(reshaped.shape)
    #print(distance.pdist(reshaped).shape)
    original=img.copy()
    for i in range(img.shape[0]):
        print(i)
        for j in range(img.shape[1]):
            print(j)
            #print("pixel")
            prev_value=img[i][j]
            while True:
                #print(prev_value)
                value=shift(prev_value, original, bandwidth)
                #print(value)
                img[i][j]=value
                if np.allclose(prev_value, value, atol=0.5, rtol=0.0):
                    #print("true")
                    break
                prev_value=value
    return img
def shift(value, pixels, bandwidth):
    shifts=np.zeros(3)
    total=0.0
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            pixel=pixels[i,j]
            distance=euclidean(pixel, value)
            #try with flat weight within range
            #if distance<bandwidth:
            #weight=(1/(math.sqrt(2*math.pi*bandwidth)))*math.exp(-distance**2/(2*bandwidth**2))
            #for k in range(3):
            #    shifts[k]+=pixel[k]*weight
            if distance<bandwidth:
                for k in range(3):
                    shifts[k]+=pixel[k]
                total+=1
    #print(shifts)
    #print(total)
    return shifts/total

def random_seed_locations(k, image):
    random.seed(0)
    centroids = np.zeros((k,2))
    for i in range(k):
        new = [0,0]
        new[0] = random.randrange(image.width)
        new[1] = random.randrange(image.height)
        #checks to avoid duplicate centroids
        while new in centroids:
            new = [0,0]
            new[0] = random.randrange(image.width)
            new[1] = random.randrange(image.height)
        centroids[i]=new
    return centroids
def random_seed_values(k, image):
    random.seed(0)
    centroids = np.zeros((k,2))
    values = np.zeros((k,3))
    for i in range(k):
        new = [0,0]
        new[0] = random.randrange(image.shape[0])
        new[1] = random.randrange(image.shape[1])
        #checks to avoid duplicate centroids
        while new in centroids:
            new = [0,0]
            new[0] = random.randrange(image.shape[0])
            new[1] = random.randrange(image.shape[1])
        centroids[i]=new
        values[i] = image[new[0],new[1]]
    return values
def probabilistic_seed_values(k, image):
    random.seed(0)
    values = np.zeros((k,3))
    new = [0, 0, 0]
    new[0] = random.randrange(256)
    new[1] = random.randrange(256)
    new[2] = random.randrange(256)
    values[0]=new
    c1 = np.ndarray.flatten(image[:,:,0])
    c2 = np.ndarray.flatten(image[:,:,1])
    c3 = np.ndarray.flatten(image[:,:,2])
    #dist to 1: the higher the more likely
    #dist to 2: same etc.
    #each iteration: calc distance to most recent
    #add to totals
    #pick (choice) with normalized dists as weights
    for i in range(1, k):
        new = [0, 0, 0]
        new[0] = np.random.choice(c1, p=np.absolute(c1-values[0][0]))
        print(new)


        #checks to avoid duplicate centroids
        #while new in centroids:
        #    new = [0,0]
        #    new[0] = random.randrange(image.shape[0])
        #    new[1] = random.randrange(image.shape[1])

    return values

def kmeans(image, k):
    '''Runs kmeans segmentation on the given image into k segments
    '''
    img=np.array(image).astype("int")
    #covariance for mahalanobis
    #reshaped=np.reshape(img, (img.shape[0]*img.shape[1],3))
    #covariance=np.cov(reshaped, rowvar=False)
    #print(covariance)
    centroid_values=probabilistic_seed_values(k,img)
    #old_means=np.zeros((len(centroid_means),5))
    old_means=np.zeros((k,3))
    #continously updates the centroids and cluster assignments
    while True:
        image_matrix, centroid_values = cluster(img, centroid_values)
        print(centroid_values-old_means)
        #accepts the image matrix if no centroid value has changed by more than 1.0
        if np.allclose(centroid_values, old_means, atol=1.0, rtol=0.0):
            return image_matrix
        old_means=centroid_values
    return image_matrix

def euclidean_color_and_distance(centroid, pixel, shape):
    h=(centroid[0]-pixel[0])**2
    s=(centroid[1]-pixel[1])**2
    v=(centroid[2]-pixel[2])**2
    y=((centroid[3]-pixel[3])*255/shape[0])**2
    x=(centroid[4]-pixel[4]*255/shape[1])**2
    color = h+s+v
    dist = x+y
    return math.sqrt(color) + math.sqrt(dist)

def euclidean_color_only(centroid, pixel):
    #need to normalize for hsv
    x=(centroid[0]-pixel[0])**2
    y=(centroid[1]-pixel[1])**2
    z=(centroid[2]-pixel[2])**2
    #technically distance is the sqrt of x+y+z, but for comparison it doesn't matter
    #dist=math.sqrt(x+y+z)
    return x+y+z

def find_closest_centroid(pixel, centroids, centroid_totals, centroid_total_counts, shape):
    '''Finds which cluster each pixel would currently belong to,
    and updates the running totals of pixel values and counts per cluster'''
    distances = []
    for value in centroids:
        #possibly mahalanobis is a better distance calculation according to research, but doesn't actually seem better and is very slow
        #distances.append(distance.mahalanobis(value, pixel, covariance))
        distances.append(euclidean_color_only(value, pixel))
    ind = np.argmin(distances)
    centroid_totals[ind]+=pixel
    centroid_total_counts[ind]+=1
    return float(ind)


def cluster(image, centroids):
    '''Takes in a matrix of pixels and one of cluster centroids, reassigns
    pixels to their new closest clusters, and updates cluster centroids.
    Returns a matrix of pixel cluster assingments and the updated centroid values
    '''
    #centroid_totals = np.zeros((len(centroids),5))
    centroid_totals = np.zeros((len(centroids),3))
    centroid_total_counts = np.zeros(len(centroids))
    image_matrix = np.zeros((len(image), len(image[0])))
    #currently it is quicker to loop and call the function rather than use numpy
    #image_matrix=np.apply_along_axis(find_closest_centroid, -1, image, centroids, centroid_totals, centroid_total_counts)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            #current=np.append(image[i][j], [i,j])
            current = image[i][j]
            image_matrix[i][j] = find_closest_centroid(current, centroids, centroid_totals, centroid_total_counts, image_matrix.shape)
    #cluster_means=np.zeros
    #currently it is quicker to loop rather than use numpy indexing
    cluster_means=centroid_totals/centroid_total_counts[:, None]
    #computes average rgb values for each cluster
    #for i in range(len(centroid_totals)):
    #    total=centroid_totals[i]
    #    sum=centroid_total_counts[i]
    #    cluster_means.append(total/sum)
    return image_matrix, cluster_means

if __name__ == "__main__":
    start=time.time()
    img = Image.open("22093.jpg")
    #HSV is potentially better than rgb, needs more testing
    img=img.convert("HSV")
    #img=Image.fromarray(mean_shift(img, 50)*200)
    img=kmeans(img, 4)
    print(img)
    img = Image.fromarray(img*30)
    stop=time.time()
    print("runtime is", stop-start)
    img.show()
