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
    # determine the facial landmarks for the face region, then convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
    for (j, (x, y)) in enumerate(shape):
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        # cv2.putText(image, str(j + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

# display the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("Output.png", image)

#For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("eyebrow", (17, 27)),
    ("eye", (36, 48)),
    ("nose", (27, 36)),
])

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
        for (x, y) in shape[i:j]:
            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

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
    # visualize all facial landmarks with a transparent overlay
    output = face_utils.visualize_facial_landmarks(image, shape)
    cv2.imshow("Image", output)
    cv2.waitKey(0)


print('Loading CLIP Model...')
model = SentenceTransformer('clip-ViT-B-32')

# Next we compute the embeddings
# To encode an image, you can use the following code:
# from PIL import Image
# encoded_image = model.encode(Image.open(filepath))
base_image_names = list(glob.glob('./images_eye_1/*.PNG'))
# base_image_names = list(glob.glob('./images_eye_2/*.jpg'))
# base_image_names = list(glob.glob('./images_eye_3/*.jpg'))
base_encoded_images = np.array([model.encode(Image.open(x)) for x in base_image_names])

images_eye = list(glob.glob('./images/eye_indoor_001.png'))

for eye in images_eye:
    eye_image = cv2.imread(eye)
    current_encoded_image = model.encode(Image.open(eye))
    ranks = model.similarity(current_encoded_image, base_encoded_images)

    i = 0
    maxNumber = 0
    maxIndex = 0
    print(ranks[0])
    for score in ranks[0]:
        if maxNumber < score:
            maxIndex = i
            maxNumber = score
        i += 1

    print(base_image_names[maxIndex])
    print("Score: {:.3f}%".format(maxNumber * 100))
