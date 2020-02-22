from thresholding import baseline_thresholding as bt
from PIL import Image
import numpy as np
import os
from collections import Counter
from parser import import_berkeley, import_weizmann
from eval import bde, region_based_eval

'''WARNING: THIS SCRIPT IS BUILT TO BE RUN ON TH CMC 307 LAB COMPUTER
AND WILL NOT WORK ON OTHERS WITHOUT CHANGING THE DIRECTORIES'''

WEIZMANN = 0
BERKELEY = 1
BDE = 2
JACCARD = 3

'''To run the two evaluation metrics on your algorithm's segmentations for a given dataset,
indicate which dataset you are running on when calling this function (WEIZMANN or BERKELEY),
and indicate which evaluation metric you are running (BDE or JACCARD),
then change the file paths and filenames in the marked spots in the code below.'''
def evaluate_all_images(dataset, metric):
    # run an evaluation metric on an algorithm's segmentation outputs for a particular dataset
    # (for example, run bde on baseline thresholding's outputs for weizmann 1 object dataset)
    # add each result as a line in a csv file of the following format described in the first line in the list
    csv_line_list = []
    if metric == BDE:
        csv_line_list = ["Segmentation file name, Ground truth file name, BDE value"]
    if metric == JACCARD:
        csv_line_list = ["Segmentation file name, Ground truth file name, Jaccard value"]
    
    if dataset == BERKELEY:
        location_segments = "thresholding_segmentations_test" # *** change to your file path **
        location_truths = "berkeley_ground_truths" # *** change to your file path ***

        files = os.listdir(location_segments)
        for segmentation_name in files:
            if not segmentation_name.startswith('.'):
                seg = Image.open(location_segments + "/" + segmentation_name)
                print("IMAGE OPENED: " + segmentation_name)
                # convert segmentation image into array of 0s and 1s (pixel values seem to have gotten distorted on google drive)
                seg = (np.array(seg) > 0).astype(int) # *** may need to change this line if you have more than two segments ***
                truth_string = segmentation_name[:-4] + ".seg"
                truth = import_berkeley(location_truths + "/" + truth_string)
                print("TRUTH IMPORTED: " + truth_string)
                if metric == BDE:
                    bde_score = bde(seg, truth)
                    print("bde is:", bde_score)
                    csv_line = segmentation_name + ", " + truth_string + ", " + str(bde_score)
                if metric == JACCARD:
                    jaccard_score = region_based_eval(truth, seg)
                    print("jaccard score is:", jaccard_score)
                    csv_line = segmentation_name + ", " + truth_string + ", " + str(jaccard_score)
                csv_line_list.append(csv_line)

    if dataset == WEIZMANN:
        location_segments = "thresholding_segmentations_2obj_test" # *** change to your file path ***
        location_truths = "Weizmann2Obj/GroundTruths" # *** change to your file path ***

        files = os.listdir(location_segments)
        all_truths = os.listdir(location_truths)

        for segmentation_name in files:
            if not segmentation_name.startswith('.'):
                seg = Image.open(location_segments + "/" + segmentation_name)
                print("IMAGE OPENED: " + segmentation_name)
                # convert segmentation image into array of 0s and 1s (pixel values seem to have gotten distorted on google drive)
                seg = (np.array(seg) > 0).astype(int) # *** may need to change this line if you have more than two segments ***
                current_truths = []
                for truth in all_truths:
                    if segmentation_name[:-4] + "_" in truth:
                        current_truths.append(truth)
                print("Ground truths for this image: ", current_truths)
                for truth_string in current_truths:
                    truth = import_weizmann(location_truths + "/" + truth_string)
                    print("TRUTH IMPORTED: ", truth_string)
                    if metric == BDE:
                        bde_score = bde(seg, truth)
                        print("bde is: ", bde_score)
                        csv_line = segmentation_name + ", " + truth_string + ", " + str(bde_score)
                    if metric == JACCARD:
                        jaccard_score = region_based_eval(truth, seg)
                        print("jaccard score is:", jaccard_score)
                        csv_line = segmentation_name + ", " + truth_string + ", " + str(jaccard_score)
                    csv_line_list.append(csv_line)

    #ideally output filename should list segmentation algorithm, eval metric, and dataset name so we can keep track of which is which
    with open('thresholding-berkeley-jaccard.csv','w') as file: # *** change to your desired output filename ***
        for line in csv_line_list:
            file.write(line)
            file.write('\n')

def segment_all_images():
    location = "Weizmann2Obj/SourceImages"
    print(os.path.expanduser(location))
    print("FOUND IT")
    files = os.listdir(location)

    for file in files:
        img = Image.open(location + '/' + file)
        temp = bt(img)
        img = Image.fromarray(temp * 100)
        img = img.convert("L")
        img.save("thresholding_segmentations_2obj_test/" + file)

if __name__ == "__main__":
    #segment_all_images()
    evaluate_all_images(BERKELEY, BDE)
