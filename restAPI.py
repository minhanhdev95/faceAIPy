# import the necessary packages
import datetime
import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import glob
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = SentenceTransformer('clip-ViT-B-32')


@dataclass
class Error:
    message: str
    code: int


@dataclass
class Result:
    position: str
    type: str


@dataclass
class ResponseData:
    filepath: str
    error: Optional[Error] = None
    result: List[Result] = field(default_factory=list)


def is_face_straight(shape):
    left_eye = shape[36]
    right_eye = shape[45]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle = np.degrees(np.arctan2(dy, dx))
    print(angle)
    return -5 <= angle <= 5


def compareImg(roi_gray, position):
    if position == 'eye':
        fileCompare = './images_eye_1/*.PNG'
    elif position == 'lips':
        fileCompare = './images_lips_1/*.PNG'

    roi_gray_image = Image.fromarray(roi_gray)
    base_image_names = list(glob.glob(fileCompare))
    base_encoded_images = np.array([model.encode(Image.open(x)) for x in base_image_names])

    current_encoded_image = model.encode(roi_gray_image)
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
    return Result(position=position, type=os.path.splitext(os.path.basename(base_image_names[maxIndex]))[0])


def cutImgEye(shape, image):
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
    cv2.imshow("ROI_EYE", roi_gray)

    return roi_gray


def cutImgLips(shape, image):
    lip_points = shape[48:68]
    # Tính toán tọa độ góc trên của hình chữ nhật
    # Tìm tọa độ góc trên bên trái và góc dưới bên phải của hình chữ nhật bao quanh môi
    x_min = np.min(lip_points[:, 0]) - 7
    y_min = np.min(lip_points[:, 1]) - 7
    x_max = np.max(lip_points[:, 0]) + 7
    y_max = np.max(lip_points[:, 1]) + 7

    # Độ dài và chiều rộng của hình chữ nhật
    width = x_max - x_min
    height = y_max - y_min

    # Vẽ hình chữ nhật bao quanh mắt
    cv2.rectangle(image, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 2)
    roi = image[y_min:y_min + height, x_min:x_min + width]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Hiển thị và lưu ảnh đen trắng
    cv2.imshow("ROI_LIPS", roi_gray)
    return roi_gray


def faceAI(image_stream):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    global roi_gray
    result: List[Result] = []
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # load the input image, resize it, and convert it to grayscale
    image = np.array(Image.open(image_stream))
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    if len(rects) == 0:
        return ResponseData(
            filepath="",
            error=Error(message="Face empty", code=400),
            result=[]
        )
    elif len(rects) > 1:
        return ResponseData(
            filepath="",
            error=Error(message="only 1 face", code=405),
            result=[]
        )
    # loop over the face detections
    for (i, rect) in enumerate(rects):

        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        if not is_face_straight(shape):
            return ResponseData(
                filepath="",
                error=Error(message="Face is not straight", code=401),
                result=[]
            )
        # for (j, (x, y)) in enumerate(shape):
        #     cv2.circle(image, (x, y), 2, (0, 0, 255), -1)  # Vẽ điểm đặc trưng
        #     cv2.putText(image, str(j+1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)  # Đánh số
        # loop over the face parts individually

        # Lấy các điểm đặc trưng 18 và 27
        roiEye = cutImgEye(shape, image)
        roiLips = cutImgLips(shape, image)

    cv2.imshow("Image with Eyes Bounding Box", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #so sánh ảnh
    result.append(compareImg(roiEye, 'eye'))
    result.append(compareImg(roiLips, 'lips'))

    return ResponseData(
        filepath=save_file(image_stream, "img"),
        result=result
    )


def save_file(fileStream, filename):
    current_time = datetime.datetime.now().strftime('%H%M%S_%d%m%y')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], current_time + '_' + filename)
    fileStream.seek(0)
    fileStream.save(filepath)
    return filepath


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        response_data = ResponseData(
            filepath="",
            error=Error(message="No File part", code=300),
            result=[]
        )
        return jsonify(response_data.__dict__), 400
    file = request.files['file']
    if file.filename == '':
        response_data = ResponseData(
            filepath="",
            error=Error(message="No selected file", code=301),
            result=[]
        )
        return jsonify(response_data.__dict__), 400
    if file:
        results = faceAI(file)
        return jsonify(results.__dict__), 400


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
