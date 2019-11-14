''' Functions to run image segmentation evaluation metrics.
    Metrics include:
        BDE (an edge-based metric)
        bipartite region matching evaluated with Jaccard measure'''

import numpy as np
import networkx as nx

def euclideanDistance(x1,y1,x2,y2):
    ''' Return the Euclidean distance between points (x1, y1) and (x2, y2)'''
    return ((x1-x2)**2+(y1-y2)**2)**(.5)

def createListOfEdgePixels(segmentation):
    ''' Create a list of pixels that are on the edge of a segment for a given
        segmentation. The input segmentation is a 2D numpy array where each
        element is the label of the segment that pixel falls in, and the output
        is an array of tuples describing the (x, y) locations of edge pixels.
        A pixel is considered 'on an edge' if at least one pixel
        on the four sides of it lies in a different segment.

        For example, suppose segmentation is the following:
        0 0 0 1 1 1
        0 0 0 1 1 1
        0 0 0 1 1 1
        0 0 0 1 1 1

        Then edge pixels are marked by X's below:
        0 0 X X 1 1
        0 0 X X 1 1
        0 0 X X 1 1
        0 0 X X 1 1

        Since outer edges of the image will always correspond to segment edges, we do
        not count these as edge pixels (unless a region is one pixel wide along an edge,
        in which case these edge pixels correspond to the inner edge of that segment),
        as they would always be matched to themselves when calculating BDE, giving a
        distance of 0 (thus not contributing anything to the metric).'''
    out = []
    height,length = segmentation.shape
    for i in range(length):
        for j in range(height):
            for x,y in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]:
                if (x>=0 and y>=0 and x<length and y<height):
                    if (segmentation[y,x] != segmentation[j,i]):
                        out.append((i,j))
                        break
    return out

def findNearestEdgePixel(pixel, edges):
    ''' Given a pixel, determine which edge pixel in the segmentation is closest
        to pixel in terms of Euclidean distance. Inputs are pixel, an (x, y)
        tuple describing the pixel's location, and edges, a list of tuples describing
        the locations of all pixels on segment edges. Returns the distance
        of the closest edge pixel from pixel (an integer).'''
    bestDistance = float("inf")
    for edge in edges:
        distance = euclideanDistance(pixel[0],pixel[1],edge[0],edge[1])
        if distance < bestDistance:
            bestDistance = distance
    return bestDistance

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
    segEdges = createListOfEdgePixels(segmentation)
    truEdges = createListOfEdgePixels(groundtruth)
    totalDistance = 0
    for edge in segEdges:
        distance = findNearestEdgePixel(edge, truEdges)
        totalDistance += distance
    for edge in truEdges:
        distance = findNearestEdgePixel(edge, segEdges)
        totalDistance += distance
    return totalDistance / (length*height*2)

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
    G = nx.Graph()
    true_nodes = []
    alg_nodes = []
    for i in range(height):
        for j in range(length):
            true_seg=str(truth[i,j])+"t"
            alg_seg=str(generated[i,j])+"a"
            if G.has_edge(true_seg, alg_seg) == False:
                G.add_edge(true_seg, alg_seg, weight=1)
            else:
                G[true_seg][alg_seg]['weight'] += 1
            if 'size' not in G.nodes[true_seg]:
                G.nodes[true_seg]['size']=1
                G.nodes[true_seg]['bipartite']=0
                true_nodes.append(true_seg)
            else:
                G.nodes[true_seg]['size']+=1
            if 'size' not in G.nodes[alg_seg]:
                G.nodes[alg_seg]['size']=1
                G.nodes[true_seg]['bipartite']=1
                alg_nodes.append(alg_seg)
            else:
                G.nodes[alg_seg]['size']+=1

    for node in true_nodes:
        for node2 in alg_nodes:
            if G.has_edge(node, node2) == False:
                G.add_edge(node, node2, weight=0)

    matching = nx.bipartite.maximum_matching(G)
    total=0
    for node in alg_nodes:
        match = matching[node]
        intersect = G[match][node]['weight']
        jaccard = intersect/(G.nodes[match]['size']+G.nodes[node]['size']-intersect)
        total+=jaccard
    return total/len(alg_nodes)

if __name__ == "__main__":
    ''' Test evaluation functions on simple segmentations'''
    # Test BDE
    testarray1 = np.asarray([[1,1,1,1,1,0,0,0],
                  [1,1,1,1,0,0,0,0],
                  [1,1,1,0,0,0,0,0],
                  [1,1,0,0,0,0,0,0]])

    testarray2 = np.asarray([[0,0,0,1,1,1,1,1],
                  [0,0,0,0,1,1,1,1],
                  [0,0,0,0,0,1,1,1],
                  [0,0,0,0,0,0,1,1]])
    print(bde(testarray1, testarray2))
    print(region_based_eval(testarray1, testarray2))

    # Test bipartite matching
    true = np.asarray([[1,1,1,2,2],
            [1,2,2,2,3],
            [1,2,4,4,3],
            [5,5,4,3,3],
            [5,5,5,5,3]])
    generation = np.asarray([[2,2,1,1,1],
                  [2,2,1,3,3],
                  [2,1,4,3,3],
                  [5,5,4,3,3],
                  [5,5,5,3,3]])

    print(bde(true, generation))
    print(region_based_eval(true, generation))
