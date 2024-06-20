# import the necessary packages
from sentence_transformers import SentenceTransformer, util
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
import glob
from PIL import Image
from collections import OrderedDict
import random as rng

#For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    # ("eyebrow", (17, 27)),
    ("eye", (36, 48)),
    # ("nose", (27, 36)),
])
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(gray, 1)
# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    # loop over the face parts individually
    for (name, (i, j)) in FACIAL_LANDMARKS_68_IDXS.items():
        # clone the original image so we can draw on it, then
        # display the name of the face part on the image
        clone = image.copy()
        cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        # loop over the subset of facial landmarks, drawing the
        # specific face part
        # for (x, y) in shape[i:j]:
        #     cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

        if (name == 'eye'):
            # Draw ellipse eye 1

            ellipse1 = cv2.fitEllipse(shape[36:42])
            ellipse2 = cv2.fitEllipse(shape[42:48])

            mask = np.full_like(image, (255, 255, 255))
            mask = cv2.ellipse(mask, ellipse1, (0, 0, 0), -1)
            mask = cv2.ellipse(mask, ellipse2, (0, 0, 0), -1)
            cv2.imshow("Mask", mask)

            result = cv2.bitwise_or(image, mask)
            cv2.imshow("result", result)
            # extract the ROI of the face region as a separate image
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = result[y - 4:y + h + 4, x - 4:x + w + 4]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

            # show the particular face part
            cv2.imshow("ROI", roi)
            cv2.imwrite(name + ".png", roi)
            cv2.imshow("Image", clone)
            cv2.waitKey(0)
# display the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("Output.png", image)







