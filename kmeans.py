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

#seeds:
# --random seeds, run multiple times and average
# **chosen seeds somehow: pick first at random, use prob function to place laters far away

#features:
# __use physical distance as well as rgb??? normalize, play with weights?
# --texture?
# __alternative color measure: HSV

#determining k:
# __his0tograms to determine k (mean shift?)
# **elbow method to determine k: run for 1<k<10, plot within-cluster sum of squares, look for "elbow"
# __silhouette method (cohesion/separation)

#misc:
# __probabilistic (expectation maximization)
# --smoothing

#distance metrics:
# **euclidean: standard root of summed squares
# __mahalanobis: more flexible, less circular? euclidean divided by squared variance


#https://spin.atomicobject.com/2015/05/26/mean-shift-clustering/
#curently prohibitively expensive, look for speed ups etc. (combine w/ kmeans by allowing merging?)
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

def random_seed_locations_and_values(k, image):
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
    #random.seed(0)
    values = np.zeros((k,3))
    #might want to switch this to geographic?
    new = [0, 0, 0]
    new[0] = random.randrange(256)
    new[1] = random.randrange(256)
    new[2] = random.randrange(256)
    values[0]=new
    #dist to 1: the higher the more likely
    #dist to 2: same etc.
    #each iteration: calc distance to most recent
    #add to totals
    #pick (choice) with normalized dists as weights
    img2=np.reshape(image, (image.shape[0]*image.shape[1],3))
    img_index=np.arange(img2.shape[0])
    total_distances=np.zeros(img2.shape[0])
    for i in range(1, k):
        distances=distance.cdist(img2, np.array([values[i-1]]))[:,0]
        total_distances+=distances
        probabilities=total_distances/np.sum(total_distances)
        #think this should work? each distance is weighted equally, higher is better
        #p should equal probabilities, normalized
        index = np.random.choice(img_index, p=probabilities)
        values[i]=img2[index]
    return values

def expectation_maximization_with_k(image, k, features='rgb', seed='random'):
    #initialize clusters w/seed values and also covariance matrices and weights (1/k for all)
    img=np.array(image).astype("int")
    #initialize means (ie centroids)
    if seed=='random':
        if features=='rgb':
            centroids=random_seed_values(k, img)
            old_means=np.zeros((k,3))
        else:
            centroids=random_seed_locations_and_values(k, img)
            old_means=np.zeros((k,5))
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
    #print(centroids)
    #print(covariances)
    #print(weights)
    #calculate conditional probabilities P(C | p) (gaussian?)
    #update params
    while True:
        image_matrix, centroids = expectation_cluster(img, centroids, covariances, weights, features)
        #accepts the image matrix if no centroid value has changed by more than 1.0
        print(centroids-old_means)
        if np.allclose(centroids, old_means, atol=1.0, rtol=0.0):
            #should this compare all changes? or just ones in most important cells?
            image_matrix=np.ndarray.argmax(image_matrix, axis=-1)
            #print(image_matrix.shape)
            return np.asfarray(image_matrix), np.asfarray(centroids)
        old_means=centroids
    return np.asfarray(image_matrix), np.asfarray(centroids)

def expectation_cluster(img, centroids, covariances, weights, features):
    centroid_totals = np.zeros((len(centroids),len(centroids[0])))
    centroid_total_counts = np.zeros(len(centroids))
    image_matrix = np.zeros((len(img), len(img[0]), len(centroids)))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if features=='rgb':
                current = img[i][j]
            else:
                current=np.append(img[i][j], [i,j])
            image_matrix[i][j] = find_centroid_expectations(current, centroids, centroid_totals, centroid_total_counts, weights, covariances, image_matrix.shape, features)
    cluster_probabilities=np.zeros(centroids.shape[0])
    cluster_means=np.zeros(centroids.shape)
    cluster_covariances=np.zeros(covariances.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cluster_probabilities+=image_matrix[i,j]
            cluster_means+=np.outer(image_matrix[i,j], img[i,j])
    #weights
    cluster_weights=cluster_probabilities/(img.shape[0]*img.shape[1])
    cluster_probabilities=np.where(cluster_probabilities==0, 1, cluster_probabilities)
    #means
    cluster_means=cluster_means/cluster_probabilities[:,None]
    #covariances
    for k in range(centroids.shape[0]):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                dist=img[i,j]-cluster_means[k]
                cluster_covariances[k]+=image_matrix[i,j,k]*np.outer(dist, dist)
        cluster_covariances[k]=cluster_covariances[k]/cluster_probabilities[k]
    return image_matrix, cluster_means

def find_centroid_expectations(pixel, centroids, centroid_totals, centroid_total_counts, weights, covariances, shape, features):
    #calculate P(Ck | x) for each k
    prob_pixel_given_clusters=np.zeros(centroids.shape[0])
    prob_pixel=0
    probabilities=np.zeros(centroids.shape[0])
    #print(centroids)
    for i in range(centroids.shape[0]):
        mean=centroids[i]
        covariance=covariances[i]
        distance=pixel-mean
        #possibly need to use logs to not lose information due to being too small
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

def expectation_maximization(image, features='rgb', seed='random'):
    image=image.convert("HSV")
    return elbow_method(image, 10, type='em')

def kmeans(image, features='rgb', seed='random'):
    image=image.convert("HSV")
    return elbow_method(image, 10)

def kmeans_with_k(image, k, features='rgb', seed='random'):
    '''Runs kmeans segmentation on the given image into k segments
    '''
    img=np.array(image).astype("int")
    #covariance for mahalanobis:
    #reshaped=np.reshape(img, (img.shape[0]*img.shape[1],3))
    #covariance=np.cov(reshaped, rowvar=False)
    #print(covariance)
    if seed=='random':
        if features=='rgb':
            centroids=random_seed_values(k, img)
            old_means=np.zeros((k,3))
        else:
            centroids=random_seed_locations_and_values(k, img)
            old_means=np.zeros((k,5))
    else:
        centroids=probabilistic_seed_values(k,img)
        old_means=np.zeros((k,3))
    #continously updates the centroids and cluster assignments
    iterations=0
    tolerance=1.0
    while True:
        image_matrix, centroids = cluster(img, centroids, features)
        #print(centroids-old_means)
        #accepts the image matrix if no centroid value has changed by more than 1.0
        if iterations>10:
            tolerance+=.1
        if np.allclose(centroids, old_means, atol=tolerance, rtol=0.0):
            return image_matrix, centroids
        old_means=centroids
        iterations+=1
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
    x=(centroid[0]-pixel[0])**2
    y=(centroid[1]-pixel[1])**2
    z=(centroid[2]-pixel[2])**2
    #technically distance is the sqrt of x+y+z, but for comparison it doesn't matter
    #dist=math.sqrt(x+y+z)
    return x+y+z

def find_closest_centroid(pixel, centroids, centroid_totals, centroid_total_counts, shape, features):
    '''Finds which cluster each pixel would currently belong to,
    and updates the running totals of pixel values and counts per cluster'''
    distances = []
    for value in centroids:
        #possibly mahalanobis is a better distance calculation according to research, but doesn't actually seem better and is very slow
        #distances.append(distance.mahalanobis(value, pixel, covariance))
        if features=='rgb':
            distances.append(euclidean_color_only(value, pixel))
        else:
            distances.append(euclidean_color_and_distance(value, pixel, shape))
    ind = np.argmin(distances)
    centroid_totals[ind]+=pixel
    centroid_total_counts[ind]+=1
    return float(ind)

def cluster(image, centroids, features):
    '''Takes in a matrix of pixels and one of cluster centroids, reassigns
    pixels to their new closest clusters, and updates cluster centroids.
    Returns a matrix of pixel cluster assingments and the updated centroid values
    '''
    #centroid_totals = np.zeros((len(centroids),5))
    centroid_totals = np.zeros((len(centroids),len(centroids[0])))
    centroid_total_counts = np.zeros(len(centroids))
    image_matrix = np.zeros((len(image), len(image[0])))
    #currently it is quicker to loop and call the function rather than use numpy
    #image_matrix=np.apply_along_axis(find_closest_centroid, -1, image, centroids, centroid_totals, centroid_total_counts)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if features=='rgb':
                current = image[i][j]
            else:
                current=np.append(image[i][j], [i,j])
            image_matrix[i][j] = find_closest_centroid(current, centroids, centroid_totals, centroid_total_counts, image_matrix.shape, features)
    #currently it is quicker to loop rather than use numpy indexing
    #need to support if a centroid count is 0
    #something is going weird here
    #for i in range(centroid_total_counts.shape[0]):
    #    if centroid_total_counts[i]==0:
    #        centroid_totals[i]=[np.nan]*len(centroid_totals[i])
    centroid_total_counts=np.where(centroid_total_counts==0, 1, centroid_total_counts)
    #print(centroid_totals)
    #print(centroid_total_counts)
    cluster_means=centroid_totals/centroid_total_counts[:, None]
    #computes average rgb values for each cluster
    #for i in range(len(centroid_totals)):
    #    total=centroid_totals[i]
    #    sum=centroid_total_counts[i]
    #    cluster_means.append(total/sum)
    #print(cluster_means)
    return image_matrix, cluster_means

def silhouette_method(image, max_k):
    #need to run some kind of maximization for score?
    scores=np.zeros(max_k)
    segmentations=[]
    img=np.array(image).astype("int")
    for k in range(2, max_k):
        print("segmenting:",k)
        segmentation=kmeans(image, k)
        segmentations.append(segmentation)
        print("scoring: ", k)
        scores[i]=silhouette_score(segmentation, img, k)
    return segmentation[np.argmax(scores)]

def elbow_method(image, max_k, type='kmeans'):
    start=time.time()
    image=np.array(image).astype("int")
    scores=np.zeros(max_k+1)
    segmentations=[0]*(max_k+1)
    for i in range(1, max_k+1):
        #print("segmenting: ", i)
        if type=='kmeans':
            segmentation, centroids=kmeans_with_k(image, i, seed='prob')
        else:
            segmentation, centroids=expectation_maximization_with_k(image, i, seed='prob')
        segmentations[i]=segmentation
        #print("evaluating: ", i)
        score=elbow_score(segmentation, image, i, centroids)
        #print(score)
        scores[i]=score
        if i>2:
            line_slope=(score-scores[1])/(i-1)
            #denom=math.sqrt((score-scores[1])**2+(i-1)**2)
            #num = max_k*scores[1]-score
            #dist1=abs((score-scores[1])*(i-1) - (i-1)*scores[i-1] + num)/denom
            #dist2=abs((score-scores[1])*(i-2) - (i-1)*scores[i-2] + num)/denom
            #if dist1<dist2:
            #    print("dist")
            #    print(i-2)
            #    return segmentations[i-2]
            if (scores[1]+(line_slope*(i-1))-scores[i-1]) < (scores[0]+(line_slope*(i-2))-scores[i-2]):
                print(i-2)
                return segmentations[i-2]
    #print(scores)
    line_slope=(scores[-1]-scores[1])/(max_k-1)
    distances=np.zeros(max_k+1)
    #https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points/39840218
    #distance from P3 perpendicular to a line drawn between P1 and P2.
    #d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
    #d=np.cross(p2-p1,p3-p1)/norm(p2-p1)
    #line = np.linalg.norm([scores[-1], max_k], [s])
    #y=scores[-1]-scores[1]
    #x=max_k-1
    #denom=math.sqrt(y**2+x**2)
    for i in range(1, max_k+1):
        corresponding=scores[1]+(line_slope*i)
        distances[i]=corresponding-scores[i]
        #distances[i]=abs((y*i-x*scores[i]+max_k*scores[1]-scores[-1]))/denom
    index=np.argmax(distances)
    #print(distances)
    #print("k=", index+1)
    print(index)
    stop=time.time()
    print(stop-start)
    return segmentations[index]

def elbow_score(segmentation, image, k, centroids):
    clusters=np.zeros(k)
    #cluster_sizes=np.zeros(k)
    for i in range(segmentation.shape[0]):
        for j in range(segmentation.shape[1]):
            distance=euclidean_color_only(centroids[int(segmentation[i,j])], image[i,j])
            clusters[int(segmentation[i,j])]+=distance
            #cluster_sizes[int(segmentation[i,j])]+=1
    #inertia=clusters/cluster_sizes
    return np.sum(clusters)

def silhouette_score(segmentation, image, k):
    #high value good
    #value=(b-a)/max{a,b}, value=0 if cluster size is 1
    #a(i)=1/(clustersize-1) * sum(dist(i,j)) for j in cluster, j!=i
    #b(i)=min(1/clustersize_j*sum(dist(i,j))) for i!=j
    flat_segs=np.ndarray.flatten(segmentation)
    flat_img=np.ndarray.reshape(image, (image.shape[0]*image.shape[1], 3))
    print(flat_img.shape)
    distances=distance.pdist(flat_img)
    cluster_sizes=np.zeros(k)
    a=np.zeros(flat_segs.shape[0])
    b=np.zeros((flat_segs.shape[0], k))
    for i in range(flat_segs.shape[0]):
        cluster_sizes[flat_segs[i]]+=1
        for j in range(flat_segs.shape[0]):
            if i!=j:
                if flat_segs[i]==flat_segs[j]:
                    a[i]+=distances[i,j]
                else:
                    b[i,flat_segs[j]]+=distances[i,j]
    value=0
    for i in range(flat_segs.shape[0]):
        a[i]=a[i]/(cluster_sizes[flat_segs[i]]-1)
        for j in range(k):
            b[i,j]=b[i,j]/cluster_sizes[k]
        min_b=np.min(b[i])
        if cluster_size[flat_segs[i]]!=0:
            value+=(min_b-a[i])/(max(min_b, a[i]))
    return value/flat_segs.shape[0]

if __name__ == "__main__":
    filenames = os.listdir("training")
    #print(filenames)
    print("filename  color  color and dist")
    for i in range(0, len(filenames), 2):
        random.seed(0)
        filename=filenames[i]
        print(filename)
        image=Image.open("training/"+filename)
        image=image.convert("HSV")
        #truth = parser.import_berkeley("training_truths/"+filename[:-3]+"seg")
        segmentation = kmeans(image)
        img = Image.fromarray(segmentation*30)
        img.show()
        #segmentation1 = kmeans_with_k(image, 4, features='rgb')[0]
        #segmentation1 = kmeans_with_k(image, 4, seed='prob')[0]
        #print(segmentation1)
        #print("1")
        #segmentation2 = kmeans_with_k(image, 4, features='dist')[0]
        #print("eval")
        #score1 = eval.bde(segmentation1, truth)
        #score2 = eval.bde(segmentation2, truth)
        #print(filename, k)
        #print(filename, score1, score2)
        #result = Image.fromarray(segmentation*30)
        #result = result.convert("RGB")
        #result.save("training_results/"+filename)

    #start=time.time()
    #img = Image.open("22093.jpg")
    #HSV is potentially better than rgb, needs more testing
    #img=img.convert("HSV")
    #img=Image.fromarray(mean_shift(img, 50)*200)
    #img=kmeans(img, 4, seed="prob")
    #img=expectation_maximization(img, 4)
    #img=elbow_method(img, 6)
    #np.set_printoptions(threshold=sys.maxsize)
    #print(img*30)
    #img = Image.fromarray(img*30)
    #stop=time.time()
    #print("runtime is", stop-start)
    #img.show()
