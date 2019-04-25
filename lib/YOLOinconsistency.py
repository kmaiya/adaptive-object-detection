'''
    This test script can be used for 4 things based on input arguments
    1). Obtain detections for a dataset of images (default)
    2). Obtain fragment error for detection of a video sequence (--test_consistency)
    3). Obtain object score sensitivity for a video sequence (--test_obj_score)
    4). Obtain the accuracy of results via TP, FP, precision and recall
'''

from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/MOT17.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.45, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
parser.add_argument("--test_consistency", type=bool, default=False, help="whether to store fragment error data in output file")
parser.add_argument("--test_obj_score", type=bool, default=True, help="whether to store objectness score error data in output file")
parser.add_argument("--outfile", type=str, default="/local/a/cam2/data/motchallenge/mot17/testprediction/", help="folder in which to store results text file")
parser.add_argument("--outfile2", type=str, default="/local/a/cam2/data/motchallenge/mot17/testobjscores/", help="folder in which to store results text file for objectness scores")
parser.add_argument("--class_conf_thres", type=float, default=0.95, help="class confidence threshold for testing low object threshold predictions")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
test_path = data_config["valid"]
num_classes = int(data_config["classes"])

# Initiate model
model = Darknet(opt.model_config_path)
model.load_weights(opt.weights_path)

if cuda:
    model = model.cuda()

model.eval()

# Get dataloader
dataset = ListDataset(test_path)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print("Compute mAP...")

all_detections = []
all_annotations = []

prev_boxes = None
prev_labels = None
pred_boxes = None
pred_labels = None

prev_class_conf = None
prev_obj_conf = None
class_conf = None
obj_conf = None

fragment_error_count = 0

lines = []
with open(test_path) as f:
    for line in f: 
        line = line.strip()
        lines.append(line)

if (opt.test_consistency):
    outfile = opt.outfile + 'pred' + data_config["valid"].split("/")[1]
    out = open(outfile, "w")

if (opt.test_obj_score):
    outfile2 = opt.outfile2 + 'pred' + data_config["valid"].split("/")[1]
    out2 = open(outfile2, "w")

img_counter = 0
img_counter_2 = 0

for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

    print('{} '.format(batch_i))
    imgs = Variable(imgs.type(Tensor))

    with torch.no_grad():
        outputs = model(imgs)
        outputs = non_max_suppression(outputs, 80, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

    for output, annotations in zip(outputs, targets):
       
        prev_boxes = pred_boxes
        prev_labels = pred_labels
        prev_class_conf = class_conf
        prev_obj_conf = obj_conf

        all_detections.append([np.array([]) for _ in range(num_classes)])
        if output is not None:
            pred_boxes = output[:, :5].cpu().numpy()
            scores = output[:, 4].cpu().numpy()
            pred_labels = output[:, -1].cpu().numpy()

            # Order by confidence
            sort_i = np.argsort(scores)
            pred_labels = pred_labels[sort_i]
            pred_boxes = pred_boxes[sort_i]

            obj_conf = output[:, 4]
            class_conf = output[:, 5]

            #analyzing all images apart from first one --batch-size = 1
            if batch_i != 0 and batch_i != 5000:

                #for each predicition in current image, getting highest matching iou from previous frame detections
                curr_ious = bbox_iou_numpy(prev_boxes[:,0:4], pred_boxes[:,0:4])
                
                #getting row wise and column wise sum for the ious
                col_iou_sum = np.sum(curr_ious, axis=0)
                row_iou_sum = np.sum(curr_ious, axis=1)

                curr_ious = np.sum(curr_ious, axis=1)
                curr_ious = np.where((curr_ious > 0.8), 1, 0)

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
                fragment_error_count += curr_img_error

                obj_score = 0

                if obj_conf.shape == prev_obj_conf.shape:            
                    v = (obj_conf.cpu().numpy() - prev_obj_conf.cpu().numpy())
                    obj_score = np.multiply(v, v)
                    obj_score = np.sum(obj_score)
                    obj_score = np.sqrt(obj_score)
                elif obj_conf.shape > prev_obj_conf.shape:
                    n = prev_obj_conf.shape[0]
                    idx = col_iou_sum.argsort()[-n:][::-1]


                    tmp_obj_conf = [obj_conf[i] for i in idx]

                    tmp_obj_conf= torch.FloatTensor(tmp_obj_conf)

                    v = (tmp_obj_conf.cpu().numpy() - prev_obj_conf.cpu().numpy())
                    obj_score = np.multiply(v, v)
                    obj_score = np.sum(obj_score)
                    obj_score = np.sqrt(obj_score)
                else:
                    n = obj_conf.shape[0]
                    idx = row_iou_sum.argsort()[-n:][::-1]

                    tmp_prev_obj_conf = [prev_obj_conf[i] for i in idx]

                    tmp_prev_obj_conf = torch.FloatTensor(tmp_prev_obj_conf)
                    v = (obj_conf.cpu().numpy() - tmp_prev_obj_conf.cpu().numpy())
                    obj_score = np.multiply(v, v)
                    obj_score = np.sum(obj_score)
                    obj_score = np.sqrt(obj_score)

                if (opt.test_obj_score):
                    path = lines[img_counter_2]
                    img_counter_2 += 1
                    k = '{} {}\n'.format(path, obj_score)
                    out2.write(k)

                if (opt.test_consistency):
                    path = lines[img_counter]
                    img_counter += 1
                    k = '{} {}\n'.format(path, curr_img_error)
                    out.write(k)

            for label in range(num_classes):
                all_detections[-1][label] = pred_boxes[pred_labels == label]

        all_annotations.append([np.array([]) for _ in range(num_classes)])
        if any(annotations[:, -1] > 0):

            annotation_labels = annotations[annotations[:, -1] > 0, 0].numpy()
            _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]

            # Reformat to x1, y1, x2, y2 and rescale to image dimensions
            annotation_boxes = np.empty_like(_annotation_boxes)
            annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
            annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
            annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
            annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
            annotation_boxes *= opt.img_size

            for label in range(num_classes):
                all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

print('Final fragment error for conf thresh: {} is : {}\n'.format(opt.conf_thres, fragment_error_count))

average_precisions = {}
with open('data/coco.names') as f:
        lines = f.read().splitlines()

accData = open("accuracy_data.txt", "w")

accuracy_data = 'Class Name                 Recall          Precision       TP              FN\n\n'
accData.write(accuracy_data)

for label in range(num_classes):
    true_positives = []
    scores = []
    num_annotations = 0

    TPup = 0
    TPdown = 0

    for i in tqdm.tqdm(range(len(all_annotations)), desc=f"Computing AP for class '{label}'"):
        detections = all_detections[i][label]
        annotations = all_annotations[i][label]

        num_annotations += annotations.shape[0]
        detected_annotations = []

        for *bbox, score in detections:
            scores.append(score)

            if annotations.shape[0] == 0:
                true_positives.append(0)
                continue

            overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap = overlaps[0, assigned_annotation]
                                                
            if max_overlap >= opt.iou_thres and assigned_annotation not in detected_annotations:
                true_positives.append(1)
                detected_annotations.append(assigned_annotation)
            else:
                true_positives.append(0)
     
    # no annotations -> AP for this class is 0
    if num_annotations == 0:
        average_precisions[label] = 0
        continue

    true_positives = np.array(true_positives)
    false_positives = np.ones_like(true_positives) - true_positives
    # sort by score
    indices = np.argsort(-np.array(scores))
    false_positives = false_positives[indices]
    true_positives = true_positives[indices]

    # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives = np.cumsum(true_positives)
     
    # compute recall and precision
    recall = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    # compute average precision
    average_precision = compute_ap(recall, precision)
    average_precisions[label] = average_precision
  
    accuracy_data = '{:16s}           {:.3f}           {:.3f}           {:.3f}           {:.3f}\n'.format(lines[label], np.mean(recall), np.mean(precision), np.mean(precision), 1 - np.mean(precision))
    accData.write(accuracy_data)

accData.close()
if (opt.test_consistency):
    out.close()
if (opt.test_obj_score):
    out2.close()

