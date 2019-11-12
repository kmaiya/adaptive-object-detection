'''
    This script can be used for calculating fragment error for the object detections for 
    one video sequence. Fragment error is a measure of consistency and stability of detections.

    The script takes in as input a path to the directory containing multiple text (.txt) files. (--dir)
    Each text file holds results for all objects detected in that image. The number of text files
    in the input directory is the number of images in the video sequence. 

    Format of 1 line of the text file: 

    %s,%f,%f,%f,%f,%f,%f\n 

    The string is the class label, and the floats are object confidence, class confidence, x1, y1, x2, y2.
    The last 4 are the bounding box coordinates of the object. The first float (object confidence) is optional
    depending on the algorithm used to generate the detections. 
'''

from __future__ import division

import os
import sys
import time
import glob
import datetime
import argparse
import tqdm
import numpy as np

'''Utility functions'''

def get_output(fname, is_conf):

    """Gets detectionn data from 1 text file.
    Parameters
    ----------
    fname : string name of text file containing detections
        
    Returns
    -------
    : list containing one ndarray:
        (num_images, 7) shaped array with IoUs
    """

    with open(fname) as f:
        data = f.readlines()

    data = [x.strip('\n').split(',') for x in data]

    num = len(data)

    outputs = np.zeros((num, 7))

    if(is_conf):
    #assign output nd-array
        for i in range(num):
            outputs[i, 0], outputs[i, 1], outputs[i, 2], outputs[i, 3], outputs[i, 4], outputs[i, 5] = float((data[i])[3]), float((data[i])[4]), float((data[i])[5]), float((data[i])[6]), float((data[i])[1]), float((data[i])[2])
    else:
        for i in range(num):
            outputs[i, 0], outputs[i, 1], outputs[i, 2], outputs[i, 3], outputs[i, 4] = float((data[i])[2]), float((data[i])[3]), float((data[i])[4]), float((data[i])[5]), float((data[i])[1])

    #convert to list
    out = []
    out.append(outputs)

    return out

def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(np.expand_dims(box1[:, 0], 1), box2[:, 0])
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(np.expand_dims(box1[:, 1], 1), box2[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih
    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih
    return intersection / ua

'''Main Script'''

#set default path for parser args for YOLO
normal = '/local/b/cam2/data/motchallenge/mot17/test/MOT17-01/original_detection_text_file/'
aa = '/local/b/cam2/data/motchallenge/mot17/train/MOT17-02/antialiased_detection_text_file/'

#set arg path for non YOLO DETECTORS
path = '/local/a/ksivaman/YOLO/PyTorch-YOLOv3/tmp_txt_files/MOT17-14/'

#get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--iou_thres", type=float, default=0.8, help="iou threshold for one-to-one pairing of images in two consecutive frames")
parser.add_argument("--conf_thres", type=float, default=0.5, help="confidence threshold value for detections in the test file")
parser.add_argument("--dir", type=str, default=normal, help="path of directory containing detection text files")
parser.add_argument("--is_obj_conf", type=bool, default=False, help="flag for whether object confidence is present or not")
opt = parser.parse_args()

prev_boxes = None
prev_labels = None
pred_boxes = None
pred_labels = None

prev_class_conf = None
prev_obj_conf = None
class_conf = None
obj_conf = None

#initializing error to 0
fragment_error_count = 0

#images = #text files in input directory
num_images = len([x for x in os.listdir(opt.dir)]) - 1

frag_errors = open('/local/b/cam2/data/motchallenge/mot17/newaa_results/mot17-10_error_per_frame.txt', 'w')

for batch in range(num_images):

    index = batch+1
    name_format = ''
    if (index < 10):
        name_format = '00000{}.txt'.format(index)
    elif (index < 100):
        name_format = '0000{}.txt'.format(index)
    elif (index < 1000):
        name_format = '000{}.txt'.format(index)
    else:
        name_format = '00{}.txt'.format(index)

    fname = opt.dir + name_format
    outputs = get_output(fname, opt.is_obj_conf)

    if (batch % 2) or (batch == 0):
        for output in outputs:
            #updating previous boxes, labels, and confidences
            prev_boxes = pred_boxes
            prev_labels = pred_labels
            prev_class_conf = class_conf
            prev_obj_conf = obj_conf

            if output is not None:

                pred_boxes = output[:, :5]
                scores = output[:, 4]
                pred_labels = output[:, -1]
                
                # Order by confidence
                sort_i = np.argsort(scores)
                pred_labels = pred_labels[sort_i]
                pred_boxes = pred_boxes[sort_i]

                obj_conf = output[:, 4]
                class_conf = output[:, 5]

                #analyzing all images apart from first one (first image has no previous frame)
                if batch != 0:

                    #for each predicition in current image, getting highest matching iou from previous frame detections
                    curr_ious = bbox_iou_numpy(prev_boxes[:,0:4], pred_boxes[:,0:4])
                    
                    #get number of detection in the image
                    num_detections = len(pred_boxes)

                    #getting row wise and column wise sum for the ious
                    col_iou_sum = np.sum(curr_ious, axis=0)
                    row_iou_sum = np.sum(curr_ious, axis=1)

                    curr_ious = np.sum(curr_ious, axis=1)
                    curr_ious = np.where((curr_ious > opt.iou_thres), 1, 0)

                    #getting the predicted x coordinated and y coordinates
                    x_coord_preds = prev_boxes[:,0]
                    y_coord_preds = prev_boxes[:,1]
                    
                    #checking lower bound and upper bound of x so that the image is not on the verge on exiting the frame
                    x_lb = x_coord_preds < 0
                    is_x_lb = np.where(x_lb == True, 1, 0)
                    x_ub = x_coord_preds > 400
                    is_x_ub = np.where(x_ub == True, 1, 0)

                    #updating iou vector
                    curr_ious = np.where(curr_ious + x_lb >= 1 , 1, 0)
                    curr_ious = np.where(curr_ious + x_ub >= 1 , 1, 0)
                    curr_ious = np.where(prev_labels == 0, curr_ious, 1)

                    #updating frament error value
                    curr_img_error = len(curr_ious) - np.sum(curr_ious)
                    frag_errors.write('{}\n'.format(curr_img_error))
                    if num_detections == 0:
                        continue
                    fragment_error_count += curr_img_error / num_detections

frag_errors.close()
fragment_error_count = (2 * fragment_error_count)/num_images
print('Final fragment error for conf thresh: {} is : {}\n'.format(opt.conf_thres, fragment_error_count))