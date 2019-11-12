from PIL import Image
import numpy as np

def import_weizmann_1(filename):
    '''converts a weizmann-segmented image with one object into a 2d matrix'''
    image = Image.open(filename)
    #image_matrix=np.zeroes(image.width, image.height)
    image_matrix=np.array(image)
    for i in range(image.height):
        for j in range(image.width):
            rgb=image_matrix[i][j]
            if (rgb[0] != rgb[1]) or (rgb[1] != rgb[2]) or (rgb[2] != rgb[0]):
                print("seg")
                image_matrix[i][j]=1
            else:
                image_matrix[i][j]=0
    print(image_matrix)
