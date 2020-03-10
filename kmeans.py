from PIL import Image
import random
import math
import numpy as np
from operator import add
from operator import sub
import time
from scipy.spatial import distance
from scipy.stats import norm
import sys
import os
import eval
import parser

#mean shift
#This runs very slowly, and is not fully tested
def mean_shift(image, bandwidth):
    ''' Mean-shift variant of k-means algorithm, which uses k=n to start with
    data point as the center of its own cluster, then merges them if they are
    similar (within the bandwidth)
    This current implementation has a very high runtime, and as such is not
    recommended or fully tested.
    Coding advice taken from https://spin.atomicobject.com/2015/05/26/mean-shift-clustering/
    '''
    img=np.array(image).astype("int")
    original=img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            prev_value=img[i][j]
            while True:
                value=shift(prev_value, original, bandwidth)
                img[i][j]=value
                if np.allclose(prev_value, value, atol=0.5, rtol=0.0):
                    break
                prev_value=value
    return img
def shift(value, pixels, bandwidth):
    '''Updates a pixel value for the mean shift algorithm based on all other pixel values
    Currently updates the pixel value with on the color distance to pixels if they are within
    the bandwidth range.
    The commented out alternative uses a weighted formula, rather than the flat distance
    '''
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
    return shifts/total

#expectation maximization
#this runs slowly
def expectation_maximization(image, seed='random'):
    '''Runs  expectation maximization through the elbow method with
    a maximum k value of 10
    '''
    return elbow_method(image, 10, seed, type='em')
def expectation_maximization_with_k(image, k, seed='random'):
    '''Expectation-maximization variant of k-means, which rather than assigning
    each pixel to a cluster at each iteration, gives the probability that each
    pixel belongs to each cluster. Runs much slower than standard k-means
    '''
    #initialize clusters w/seed values and also covariance matrices and weights (1/k for all)
    img=np.array(image).astype("int")
    #initialize means (ie centroids)
    if seed=='random':
        centroids=random_seed_values(k, img)
        old_means=np.zeros((k,3))
    else:
        centroids=probabilistic_seed_values(k,img)
        old_means=np.zeros((k,3))
    #initialize covariances (as identity matrices)
    covariances=np.zeros((k, centroids.shape[1],centroids.shape[1]))
    for i in range(k):
        covariance=np.identity(centroids.shape[1])
        covariances[i]=covariance
    #initialize weights
    weights=np.full(k, 1.0/k)
    while True:
        image_matrix, centroids = expectation_cluster(img, centroids, covariances, weights)
        #accepts the image matrix if no centroid value has changed by more than 1.0
        print(centroids-old_means)
        if np.allclose(centroids, old_means, atol=1.0, rtol=0.0):
            image_matrix=np.ndarray.argmax(image_matrix, axis=-1)
            return np.asfarray(image_matrix), np.asfarray(centroids)
        old_means=centroids
    return np.asfarray(image_matrix), np.asfarray(centroids)
def expectation_cluster(img, centroids, covariances, weights):
    '''Updates the image matrix with the new probabilities of belonging to each
    cluster for each pixel, then updates each clusters mean, weight, and covariance
    matrix.
    '''
    centroid_totals = np.zeros((len(centroids),len(centroids[0])))
    centroid_total_counts = np.zeros(len(centroids))
    image_matrix = np.zeros((len(img), len(img[0]), len(centroids)))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            current = img[i][j]
            #updates pixel probabilities
            image_matrix[i][j] = find_centroid_expectations(current, centroids, centroid_totals, centroid_total_counts, weights, covariances, image_matrix.shape)
    cluster_probabilities=np.zeros(centroids.shape[0])
    cluster_means=np.zeros(centroids.shape)
    cluster_covariances=np.zeros(covariances.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cluster_probabilities+=image_matrix[i,j]
            cluster_means+=np.outer(image_matrix[i,j], img[i,j])
    #updates cluster weights
    cluster_weights=cluster_probabilities/(img.shape[0]*img.shape[1])
    cluster_probabilities=np.where(cluster_probabilities==0, 1, cluster_probabilities)
    #updates cluster means
    cluster_means=cluster_means/cluster_probabilities[:,None]
    #updates cluster covariances
    for k in range(centroids.shape[0]):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                dist=img[i,j]-cluster_means[k]
                cluster_covariances[k]+=image_matrix[i,j,k]*np.outer(dist, dist)
        cluster_covariances[k]=cluster_covariances[k]/cluster_probabilities[k]
    return image_matrix, cluster_means
def find_centroid_expectations(pixel, centroids, centroid_totals, centroid_total_counts, weights, covariances, shape):
    ''' Calculates the probability of belonging to each cluster for the given pixel
    '''
    prob_pixel_given_clusters=np.zeros(centroids.shape[0])
    prob_pixel=0
    probabilities=np.zeros(centroids.shape[0])
    for i in range(centroids.shape[0]):
        mean=centroids[i]
        covariance=covariances[i]
        distance=pixel-mean
        exponent=np.matmul(np.matmul(np.transpose(distance),np.linalg.inv(covariance)), distance)
        prob_pixel_given_cluster=math.exp(-.5*exponent)/math.sqrt(2*math.pi*np.linalg.det(covariance))
        prob_pixel_given_clusters[i]=prob_pixel_given_cluster
        prob_pixel+=prob_pixel_given_cluster/weights[i]
    for i in range(centroids.shape[0]):
        if prob_pixel==0:
            prob_cluster=0
        else:
            prob_cluster=prob_pixel_given_clusters[i]*weights[i]/prob_pixel
        probabilities[i]=prob_cluster
    return probabilities

#seeding
def random_seed_locations_and_values(k, image):
    '''Randomly picks k locations with the image, and returns the selected pixel
    values and locations in the form [[r,g,b,x,y], ...]
    '''
    locations = np.zeros((k,2))
    centroids = np.zeros((k,5))
    for i in range(k):
        new = [0,0]
        new[0] = random.randrange(image.shape[0])
        new[1] = random.randrange(image.shape[1])
        #checks to avoid duplicate centroids
        while new in locations:
            new = [0,0]
            new[0] = random.randrange(image.shape[0])
            new[1] = random.randrange(image.shape[1])
        locations[i]=new
        centroids[i]=np.append(image[new[0],new[1]], [new[0],new[1]])
    return centroids
def random_seed_values(k, image):
    ''' Randomly picks k locations with the image, and returns the selected pixel
    values in the form [[r,g,b], ...]
    '''
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
    '''Probabilistically selects k seeds of pixel values: the first seed is
    chosen randomly, then later selections are weighted to prefer values further
    away from existing seeds.
    Returns values in the form [[r,g,b], ...]
    '''
    values = np.zeros((k,3))
    new = [0, 0, 0]
    #randomly selects first seed
    new[0] = random.randrange(256)
    new[1] = random.randrange(256)
    new[2] = random.randrange(256)
    values[0]=new
    img2=np.reshape(image, (image.shape[0]*image.shape[1],3))
    img_index=np.arange(img2.shape[0])
    total_distances=np.zeros(img2.shape[0])
    for i in range(1, k):
        #finds distances from all pixel to the most recent seed
        distances=distance.cdist(img2, np.array([values[i-1]]))[:,0]
        #adds latest distance onto running total distance
        total_distances+=distances
        probabilities=total_distances/np.sum(total_distances)
        #uses distances to weight seed selection
        index = np.random.choice(img_index, p=probabilities)
        values[i]=img2[index]
    return values

#distance metric
def euclidean_color_only(centroid, pixel):
    '''Calculates the euclidean distance in rgb color space from a pixel to a centroid
    '''
    x=(centroid[0]-pixel[0])**2
    y=(centroid[1]-pixel[1])**2
    z=(centroid[2]-pixel[2])**2
    #technically distance is the sqrt of x+y+z, but for comparison it doesn't matter
    return x+y+z

#choosing k
def elbow_method(image, max_k, seed, type='kmeans'):
    '''Runs algorithm with increasing values of k, up to the maximum.
    For each result, calculates the sum-of-squared-error, and finds the "elbow"
    where additional clusters don't signifacntly decrease the in-cluster variance
    '''
    start=time.time()
    image=np.array(image).astype("int")
    scores=np.zeros(max_k+1)
    segmentations=[0]*(max_k+1)
    for i in range(1, max_k+1):
        if type=='kmeans':
            segmentation, centroids=kmeans_with_k(image, i, seed)
        else:
            segmentation, centroids=expectation_maximization_with_k(image, i, seed)
        segmentations[i]=segmentation
        score=elbow_score(segmentation, image, i, centroids)
        scores[i]=score
        if i>2:
            line_slope=(score-scores[1])/(i-1)
            if (scores[1]+(line_slope*(i-1))-scores[i-1]) < (scores[0]+(line_slope*(i-2))-scores[i-2]):
                return segmentations[i-2]
    line_slope=(scores[-1]-scores[1])/(max_k-1)
    distances=np.zeros(max_k+1)
    for i in range(1, max_k+1):
        corresponding=scores[1]+(line_slope*i)
        distances[i]=corresponding-scores[i]
    index=np.argmax(distances)
    stop=time.time()
    return segmentations[index]
def elbow_score(segmentation, image, k, centroids):
    '''Calculates the total sum-of-squared-error, ie within-cluster-variance,
    for the given segmentation
    '''
    clusters=np.zeros(k)
    for i in range(segmentation.shape[0]):
        for j in range(segmentation.shape[1]):
            distance=euclidean_color_only(centroids[int(segmentation[i,j])], image[i,j])
            clusters[int(segmentation[i,j])]+=distance
    return np.sum(clusters)

#kmeans
def kmeans(image, seed='random'):
    '''Runs k-means through the elbow method with a maximum k value of 10
    '''
    return elbow_method(image, 10, seed, type='kmeans')
def kmeans_with_k(image, k, seed='random'):
    '''Runs kmeans segmentation on the given image into k segments
    '''
    img=np.array(image).astype("int")
    if seed=='random':
        centroids=random_seed_values(k, img)
        old_means=np.zeros((k,3))
    else:
        centroids=probabilistic_seed_values(k,img)
        old_means=np.zeros((k,3))
    #continously updates the centroids and cluster assignments
    iterations=0
    tolerance=1.0
    while True:
        image_matrix, centroids = cluster(img, centroids)
        #accepts the image matrix if no centroid value has changed by more than 1.0
        #if the number of iterations is above ten, start gradually incrementing
        #the tolerance to avoid very long runtimes
        if iterations>10:
            tolerance+=.1
        if np.allclose(centroids, old_means, atol=tolerance, rtol=0.0):
            return image_matrix, centroids
        old_means=centroids
        iterations+=1
    return image_matrix
def find_closest_centroid(pixel, centroids, centroid_totals, centroid_total_counts, shape):
    '''Finds which cluster each pixel would currently belong to,
    and updates the running totals of pixel values and counts per cluster'''
    distances = []
    for value in centroids:
        distances.append(euclidean_color_only(value, pixel))
    #index of closest cluster center
    ind = np.argmin(distances)
    centroid_totals[ind]+=pixel
    centroid_total_counts[ind]+=1
    return float(ind)
def cluster(image, centroids):
    '''Takes in a matrix of pixels and one of cluster centroids, reassigns
    pixels to their new closest clusters, and updates cluster centroids.
    Returns a matrix of pixel cluster assingments and the updated centroid values
    '''
    centroid_totals = np.zeros((len(centroids),len(centroids[0])))
    centroid_total_counts = np.zeros(len(centroids))
    image_matrix = np.zeros((len(image), len(image[0])))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            current = image[i][j]
            image_matrix[i][j] = find_closest_centroid(current, centroids, centroid_totals, centroid_total_counts, image_matrix.shape)
    centroid_total_counts=np.where(centroid_total_counts==0, 1, centroid_total_counts)
    cluster_means=centroid_totals/centroid_total_counts[:, None]
    return image_matrix, cluster_means
