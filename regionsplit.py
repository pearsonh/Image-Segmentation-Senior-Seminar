from PIL import Image
import numpy as np
import statistics as stat

def regionSplitAndMerge(image, threshold):
    '''Base function for split and merge.
        Takes a PIL image and an integer as input, and outputs a numpy array
        of the segmentation.
        After processing the input image into a numpy array greyscale representation
        it runs splitAndMerge to produce the segmented array'''
    imageGrey = image.convert(mode='L')
    img = np.array(imageGrey).astype("int")
    blank = np.empty(img.shape)
    splitMergeSeg = splitAndMerge(img, blank, threshold)
    return splitMergeSeg

def splitAndMerge(img, blank, threshold):
    '''Takes a numpy array, a blank numpy array of the same size, and an integer as input.
        Outputs a segmented array using the blank numpy array.
        Recursively tests quadrants of the image to test if they are homogenous regions.
        If not, the regions are split and the test is run on each. If they are, then
        the regions are painted an average color.

        As they recurse back up the stack, collections of two or three adjacent quadrants are
        repainted to fix any oversegmenting done in the split phase.'''
    tL = img[:img.shape[0]//2, :img.shape[1]//2]
    tR = img[:img.shape[0]//2, img.shape[1]//2:]
    bL = img[img.shape[0]//2:, :img.shape[1]//2]
    bR = img[img.shape[0]//2:, img.shape[1]//2:]
    if flatTest(tL, threshold) and flatTest(tR, threshold) and flatTest(bL, threshold) and flatTest(bR, threshold):
        x,y = img.shape
        avgVal = np.mean(img)
        for i in range(x):
            for j in range(y):
                blank[i][j] = avgVal
    else:
        bHeight = blank.shape[1]
        bWidth = blank.shape[0]
        splitAndMerge(tL, blank[:bWidth//2, :bHeight//2], threshold)
        splitAndMerge(tR, blank[:bWidth//2, bHeight//2:], threshold)
        splitAndMerge(bL, blank[bWidth//2:, :bHeight//2], threshold)
        splitAndMerge(bR, blank[bWidth//2:, bHeight//2:], threshold)
        '''This section controls the merging of all regions within a quadrant'''
        if flatTest(np.concatenate([tL, tR], axis=1), threshold):
            if flatTest(np.concatenate([tL, bL], axis=0), threshold):
                blank[:bWidth//2, :bHeight//2] = np.full(tL.shape, np.mean(np.concatenate([tL, tR], axis=1)))
                blank[:bWidth//2, bHeight//2:] = np.full(tR.shape, np.mean(np.concatenate([tL, tR], axis=1)))
                blank[bWidth//2:, :bHeight//2] = np.full(bL.shape, np.mean(np.concatenate([tL, tR], axis=1)))
            else:
                blank[:bWidth//2, :bHeight//2] = np.full(tL.shape, np.mean(np.concatenate([tL, tR], axis=1)))
                blank[:bWidth//2, bHeight//2:] = np.full(tR.shape, np.mean(np.concatenate([tL, tR], axis=1)))
        if flatTest(np.concatenate([bL, bR], axis=1), threshold):
            if flatTest(np.concatenate([tR, bR], axis=0), threshold):
                blank[bWidth//2:, :bHeight//2] = np.full(bL.shape, np.mean(np.concatenate([bL, bR], axis=1)))
                blank[bWidth//2:, bHeight//2:] = np.full(bR.shape, np.mean(np.concatenate([bL, bR], axis=1)))
                blank[:bWidth//2, bHeight//2:] = np.full(tR.shape, np.mean(np.concatenate([bL, bR], axis=1)))
            else:
                blank[bWidth//2:, :bHeight//2] = np.full(bL.shape, np.mean(np.concatenate([bL, bR], axis=1)))
                blank[bWidth//2:, bHeight//2:] = np.full(bR.shape, np.mean(np.concatenate([bL, bR], axis=1)))
        elif flatTest(np.concatenate([tR, bR], axis=0), threshold):
            blank[:bWidth//2, bHeight//2:] = np.full(tR.shape, np.mean(np.concatenate([tR, bR], axis=0)))
            blank[bWidth//2:, bHeight//2:] = np.full(bR.shape, np.mean(np.concatenate([tR, bR], axis=0)))
        elif flatTest(np.concatenate([tL, bL], axis=0), threshold):
            blank[:bWidth//2, :bHeight//2] = np.full(tL.shape, np.mean(np.concatenate([tL, bL], axis=0)))
            blank[bWidth//2:, :bHeight//2] = np.full(bL.shape, np.mean(np.concatenate([tL, bL], axis=0)))
    return blank


def flatTest(array, threshold):
    '''Takes an np.array and a threshold, and outputs TRUE if the contents of the array have a
        statistical variance lower than the given threshold, and false if it does not

        This is the P predicate described in the paper; its function is to serve as the basis for whether
        or not to split, and whether or not to merge, a given region or two regions'''
    x, y = array.shape
    mean = []
    threshold = threshold
    for j in range(x):
        for k in range(y):
            mean.append(array[j,k])
    if len(mean) <= 4 or stat.variance(mean, None) <= threshold:
        return True
    else:
        return False
        
if __name__ == '__main__':
    img = Image.open("22093.jpg")
    array = regionSplitAndMerge(img, 450)
    splitAndMerge = array
    newImage = Image.fromarray(splitAndMerge)
    newImage.show()
