# import the necessary packages
from sentence_transformers import SentenceTransformer
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import glob
from PIL import Image



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
    for (j, (x, y)) in enumerate(shape):
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)  # Vẽ điểm đặc trưng
    #     cv2.putText(image, str(j+1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)  # Đánh số
    # loop over the face parts individually

    # Lấy các điểm đặc trưng 18 và 27
    point_18 = shape[17]  # Điểm thứ 18 (index 17)
    point_27 = shape[26]  # Điểm thứ 27 (index 26)
    point_30 = shape[28]  # Điểm dưới mũi
    # Tính toán tọa độ góc trên của hình chữ nhật
    top_left = point_18
    top_right = point_27
    bottom_y = point_30[1]

    # Vẽ hình chữ nhật bao quanh mắt
    cv2.rectangle(image, (top_left[0], top_left[1]), (top_right[0], bottom_y), (0, 255, 0), 2)
    roi = image[top_left[1]:bottom_y, top_left[0]:top_right[0]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Hiển thị và lưu ảnh đen trắng
    cv2.imshow("ROI", roi_gray)
    cv2.imwrite("ROI_gray.png", roi_gray)

    # Hiển thị hình ảnh kết quả
    cv2.imshow("Image with Eyes Bounding Box", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


model = SentenceTransformer('clip-ViT-B-32')
base_image_names = list(glob.glob('./images_eye_1/*.PNG'))
base_encoded_images = np.array([model.encode(Image.open(x)) for x in base_image_names])
images_eye = list(glob.glob('ROI_gray.png'))

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





