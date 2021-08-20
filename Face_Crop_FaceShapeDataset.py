##### 1. face_alignment #####
import cv2
import mediapipe as mp
import os
from tqdm import tqdm
from time import sleep
import pandas as pd
import numpy as np
import dlib
import natsort

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYES = list(range(36, 48))

abs_path = 'C:/Users/user/Desktop/Face_Class/FaceShape Dataset'

# 테스트
# dataset_paths = [
#     './testing_set/Heart/', './testing_set/Oblong/', './testing_set/Oval/',
#     './testing_set/Round/', './testing_set/Square/']
# 트레인
dataset_paths = [
    './training_set/Heart/', './training_set/Oblong/', './training_set/Oval/',
    './training_set/Round/', './training_set/Square/']

# 테스트 크롭 저장
# output_paths = [
#     './testing_set_cropped/Heart/', './testing_set_cropped/Oblong/', './testing_set_cropped/Oval/',
#     './testing_set_cropped/Round/', './testing_set_cropped/Square/']
# 트레인 크롭 저장
output_paths = [
    './training_set_cropped/Heart/', './training_set_cropped/Oblong/', './training_set_cropped/Oval/',
    './training_set_cropped/Round/', './training_set_cropped/Square/']

predictor_file = './model/shape_predictor_68_face_landmarks.dat'
MARGIN_RATIO = 1.5
OUTPUT_SIZE = (300, 300)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)


def getFaceDimension(rect):
    return (rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top())


def getCropDimension(rect, center):
    width = rect.right() - rect.left()
    half_width = (width // 2)
    (centerX, centerY) = center
    startX = (centerX - half_width) - int(20)
    endX = (centerX + half_width) + int(20)
    startY = rect.top() - int(20)
    endY = rect.bottom() + int(20)
    return (startX, endX, startY, endY)


for (path_i, dataset_path) in enumerate(dataset_paths):
    output_path = output_paths[path_i]
    os.chdir(f'{abs_path}{dataset_path[1:]}')
    # For static images:
    IMAGE_FILES = []
    files_tmp = os.listdir()
    files = natsort.natsorted(files_tmp)

    for file in files:
        f = cv2.imread(file)
        IMAGE_FILES.append(f)

    for idx, file in enumerate(IMAGE_FILES):
        image = file
        # get RGB image from BGR, OpenCV format
        image_origin = image.copy()

        (image_height, image_width) = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):
            (x, y, w, h) = getFaceDimension(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            points = np.matrix([[p.x, p.y]
                               for p in predictor(gray, rect).parts()])
            show_parts = points[EYES]

            right_eye_center = np.mean(points[RIGHT_EYE], axis=0).astype("int")
            left_eye_center = np.mean(points[LEFT_EYE], axis=0).astype("int")

            eye_delta_x = right_eye_center[0, 0] - left_eye_center[0, 0]
            eye_delta_y = right_eye_center[0, 1] - left_eye_center[0, 1]
            degree = np.degrees(np.arctan2(eye_delta_y, eye_delta_x)) - 180

            eye_distance = np.sqrt((eye_delta_x ** 2) + (eye_delta_y ** 2))
            aligned_eye_distance = left_eye_center[0,
                                                   0] - right_eye_center[0, 0]
            scale = aligned_eye_distance / eye_distance

            eyes_center = (int(left_eye_center[0, 0] + right_eye_center[0, 0]) // 2,
                           int(left_eye_center[0, 1] + right_eye_center[0, 1]) // 2)

            metrix = cv2.getRotationMatrix2D(eyes_center, degree, scale)

            warped = cv2.warpAffine(image_origin, metrix, (image_width, image_height),
                                    flags=cv2.INTER_CUBIC)

            (startX, endX, startY, endY) = getCropDimension(rect, eyes_center)

            croped = warped[startY:endY, startX:endX]
            try:
                output = cv2.resize(croped, OUTPUT_SIZE)
            except:
                pass
            #output = warped[startY:endY, startX:endX]

            output_file = f'{abs_path}{output_paths[path_i][1:]}{str(idx+1)}.jpg'
            print(output_file)
            #cv2.imshow(output_file, output)
            cv2.imwrite(output_file, output)
    print(f"{path_i+1} 번째 데이터셋 크롭완료")