from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

argp = argparse.ArgumentParser()
argp.add_argument("-i", "--image_dir", required=True)
args = vars(argp.parse_args())

folders = ["00081_stuff"]
for folder in folders:
    imageLoc = args["image_dir"] + folder + "/image-frames/"
    saveLoc = args["image_dir"] + folder + "/hog_predictions/"
    try:
        os.makedirs(saveLoc)
        print("Save Location does not exist: creating")
    except FileExistsError:
        print("Directory Exists: Continuing")
    count = 0
    for imagePath in paths.list_images(imageLoc):
        image = cv2.imread(imagePath)
        #image = imutils.resize(image, width=min(400, image.shape[1]))
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        (rects, weight) = hog.detectMultiScale(gray_image, winStride=(4,4), padding=(8,8), scale=1.05)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in  rects])
        sup = non_max_suppression(rects, probs=None, overlapThresh=.3)

        for i, (x, y, w, h) in enumerate(sup):
            cv2.rectangle(image, (x,y), (w, h), (0, 255, 0), 2)
        cv2.imwrite(saveLoc + str(count).zfill(5) + "_det.jpg", image)
        count = count + 1

