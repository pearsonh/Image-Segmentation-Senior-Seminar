from PIL import Image
import random
import math
import numpy as np
from operator import add
from operator import sub

def kmeans(image, k):
    centroids = []
    for i in range(k):
        new = [0,0]
        new[0] = random.randrange(image.width)
        new[1] = random.randrange(image.height)
        while new in centroids:
            new[0] = random.randrange(image.width)
            new[1] = random.randrange(image.height)
        centroids.append(new)
    centroid_means = []
    for centroid in centroids:
        centroid_means.append(image.getpixel((centroid[0], centroid[1])))
    old_means=np.zeros((len(centroid_means),3))
    while True:
        image_matrix, centroid_means = cluster(image, centroid_means)
        changed=False
        #print(centroid_means)
        for i in range(len(centroid_means)):
            #print(centroid_means[i])
            differences=list(map(sub,centroid_means[i],old_means[i]))
            print(differences)
            if differences!=[0.0,0.0,0.0]:
                #print("changed")
                changed=True
        if not changed:
            return image_matrix
        old_means=centroid_means
    return image_matrix

def cluster(image, centroids):
    centroid_totals = np.zeros((len(centroids),4))
    image_matrix = np.zeros((image.width, image.height))
    for i in range(image.width):
        for j in range(image.height):
            rgb = image.getpixel((i,j))
            distances = []
            for value in centroids:
                x=(value[0]-rgb[0])**2
                y=(value[1]-rgb[1])**2
                z=(value[2]-rgb[2])**2
                dist=math.sqrt(x+y+z)
                distances.append(dist)
            ind = np.argmin(distances)
            image_matrix[i][j] = ind
            totals = centroid_totals[ind]
            totals = list(map(add, totals,list(rgb)+[1]))
            centroid_totals[ind]=totals
            #totals[1] += 1
    cluster_means = []
    for total in centroid_totals:
        #print(total)
        cluster_means.append([total[0]/total[3], total[1]/total[3], total[2]/total[3]])
    #print(cluster_means)
    return image_matrix, cluster_means



#image = Image.open("1obj/100_0109/src_color/100_0109.png")
#np.set_printoptions(threshold=np.inf)
#print(kmeans(image, 2))
