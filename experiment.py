'''Command line tool to run the Image Segmentation Senior Composition project.
This tool takes in a set of commands, and currently uses it to display the images
and their segmentations.

To be added:
A full experiment as decided on by the group
Slight restructuring of commands to make things easier


'''

from PIL import Image
from kmeans import kmeans
from thresholding import baseline_thresholding
from eval import bde, region_based_eval
import argparse



def parse_args():
    '''
    Parse command line for user commands
    '''
    p = argparse.ArgumentParser()
    p.add_argument("--kmeans", action="store_true", help="Set this flag to run"+\
        "K-means image segmentation")
    p.add_argument("--segments", type=int, default=2, help="Number of segments"+\
        "for the K-means algorithm")
    p.add_argument("--threshold", action="store_true", help="Set this flag to" +\
        "run thresholding image segmentation")
    p.add_argument("--image", help="Name of image to be segmented",
        default="22093.jpg")
    p.add_argument("--displayOriginal", action="store_true",  help="Displays" +\
        "the original image along with the new one")
    p.add_argument("--evaluate", action="store_true", help="Determines whether" +\
        "evaluation metrics will be run on segmentations.")
    return p.parse_args()

def merge(imgL, imgR):
    '''A function that takes in two images and merges them to display
    the new images next to eachother:
    input: two PIL images
    output: one horizontally merged PIL image'''

    imgSize = imgL.size
    imgWidth = 2 * imgSize[0]
    imgHeight = imgSize[1]
    imgNew = Image.new('RGB', (imgWidth, imgHeight))
    imgNew.paste(imgL,(0,0))
    imgNew.paste(imgR,(int(imgWidth/2), 0))
    return imgNew


def main():
    '''Runs the specified commands from the command line using our built in
    segmentation algorithms and files. Runs on a default image for debugging
    '''
    args = parse_args()
    if args is None:
        print("Please enter arguments to run different segmentations")
    if args.kmeans:
        orig = Image.open(args.image)
        temp = kmeans(orig, args.segments)
        img = Image.fromarray(temp*(255 / args.segments))
        if args.displayOriginal:
            img = merge(orig, img)
        img.show()
    if args.threshold:
        orig = Image.open(args.image)
        temp = baseline_thresholding(orig)
        img = Image.fromarray(temp*255)
        if args.displayOriginal:
            img = merge(orig, img)
        img.show()

if __name__ == "__main__":
    main()
