from thresholding import baseline_thresholding as bt
from PIL import Image
import numpy as np
import os
from collections import Counter
from parser import import_berkeley, import_weizmann
from eval import bde, region_based_eval
import kmeans

WEIZMANN = 0
BERKELEY = 1
BDE = 2
JACCARD = 3

'''To run the two evaluation metrics on your algorithm's segmentations for a given dataset,
indicate which dataset you are running on when calling this function (WEIZMANN or BERKELEY),
indicate which evaluation metric you are running (BDE or JACCARD),
indicate the file paths of segmentations to compare and the filename you would like for your
output CSV file .'''
def evaluate_all_images(dataset, metric, location_segments, location_truths, output_filename):
    # run an evaluation metric on an algorithm's segmentation outputs for a particular dataset
    # (for example, run bde on baseline thresholding's outputs for weizmann 1 object dataset)
    #ideally output filename should list segmentation algorithm, eval metric, and dataset name
    # add each result as a line in a csv file of the following format described in the first line in the list
    csv_line_list = []
    if metric == BDE:
        csv_line_list = ["Segmentation file name, Ground truth file name, BDE value"]
    if metric == JACCARD:
        csv_line_list = ["Segmentation file name, Ground truth file name, Jaccard value"]

    if dataset == BERKELEY:
        files = os.listdir(location_segments)
        for segmentation_name in files:
            if not segmentation_name.startswith('.'):
                seg = Image.open(location_segments + "/" + segmentation_name)
                print("IMAGE OPENED: " + segmentation_name)
                seg = np.array(seg)
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
        files = os.listdir(location_segments)
        all_truths = os.listdir(location_truths)

        for segmentation_name in files:
            if not segmentation_name.startswith('.'):
                seg = Image.open(location_segments + "/" + segmentation_name)
                print("IMAGE OPENED: " + segmentation_name)
                seg = np.array(seg)
                current_truths = []
                for truth in all_truths:
                    if segmentation_name[:-4] in truth:
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

    with open(output_filename,'w') as file:
        for line in csv_line_list:
            file.write(line)
            file.write('\n')

'''Segment all images in the indicated location folder using the indicated algorithm,
and save segmentations in indicated output_folder.'''
def segment_all_images(location, algorithm, output_folder):
    print(os.path.expanduser(location))
    print("FOUND FILES")
    files = os.listdir(location)

    for file in files:
        img = Image.open(location + '/' + file)
        temp=algorithm(img)
        img = Image.fromarray(temp * 30)
        img = img.convert("L")
        img.save(output_folder + file)

if __name__ == "__main__":
    segment_all_images("Weizmann2Obj/SourceImages",bt, "thresholding_segmentations_2obj_test/")
    evaluate_all_images(WEIZMANN, BDE, "thresholding_segmentations_2obj_test","Weizmann2Obj/GroundTruths",'thresholding-weizmann-bde.csv')
