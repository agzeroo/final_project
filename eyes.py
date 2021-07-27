import cv2
import mediapipe as mp
import os
from tqdm import tqdm
from time import sleep

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

path = 'C:/final_project/final_project/data/'  # 폴더 경로
os.chdir(path)  # 해당 폴더로 이동

# For static images:
IMAGE_FILES = []

files = os.listdir(path)
for file in files:
    if '.png' in file:
        f = cv2.imread(file)
        # 이미지 256x256 resize
        # f = cv2.resize(f, dsize=(800, 800), interpolation=cv2.INTER_AREA)
        IMAGE_FILES.append(f)
    if '.jpg' in file:
        f = cv2.imread(file)
        # f = cv2.resize(f, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        IMAGE_FILES.append(f)
# print(IMAGE_FILES)

# 진행률
# for i in tqdm(range(1, 600)):
#     sleep(0.01)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
    for idx, file in enumerate(IMAGE_FILES):
        image = file
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            # print('face_landmarks:', face_landmarks)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
        # cv2.imwrite('./tmp/annotated_image_' +
        #             str(idx) + '.png', annotated_image)
        # if not cv2.imwrite('./tmp/annotated_image_' + str(idx) + '.png', annotated_image):
        #     raise Exception("Could not write image")

# 얼굴 랜드마크 468개 순서체크 이미지 생성
# print(results.multi_face_landmarks[0].landmark[0].y)


# =============================================================================================================================
'''
-눈-
큰눈 - 왼쪽 + 오른쪽 눈면적(타원)/얼굴면적(타원) 
작은눈 - 왼쪽 + 오른쪽 눈면적(타원)/얼굴면적(타원) 
찢어진눈 - 세로길이/가로길이 비율
왕방울눈 - 세로길이/가로길이 비율

-눈꼬리-
위로  - 기울기
아래로 - 기울기

'''

# =============================================================================================================================

# 눈 포인트
points = [34, 161, 145, 160, 146, 159, 154, 156, 383, 385, 381, 386, 375, 387, 374, 264, 235, 448, 11, 153]
# 얼굴 면적용 포인트


# 왼쪽 눈썹 포인트 좌, 위, 아래, 우
Eyeline_l = [34, 161, 145, 160, 146, 159, 154, 156]

# 오른쪽 눈썹 포인트 좌, 위, 아래, 우
Eyeline_r = [383, 385, 381, 386, 375, 387, 374, 264]

# 얼굴면적(가로좌, 가로우, 세로위, 세로아래)
Face_area = [235, 448, 11, 153]

# -------------------------------
# 눈
# 왼쪽
Eyeline_l_dict = {}

# 오른쪽
Eyeline_r_dict = {}

# 얼굴면적용(가로좌, 가로우, 세로위, 세로아래)
Face_area_dict = {}

# -------------------------------

# for 문
num = 1

for face in results.multi_face_landmarks:
    for (i, landmark) in enumerate(face.landmark):
        x = landmark.x
        y = landmark.y
        z = landmark.z

        shape = image.shape
        relative_x = int(x * shape[1])
        relative_y = int(y * shape[0])

        # print(relative_x, relative_y)

        cv2.circle(image, (relative_x, relative_y),
                   radius=1, color=(0, 255, 0), thickness=1)

        # -------------------------------
        # 눈
        if num in Eyeline_l:
            Eyeline_l_dict[f'{num}'] = relative_x, relative_y, x, y, z

        elif num in Eyeline_r:
            Eyeline_r_dict[f'{num}'] = relative_x, relative_y, x, y, z

        elif num in Face_area:
            Face_area_dict[f'{num}'] = relative_x, relative_y, x, y, z

        # -------------------------------

        num += 1

        cv2.putText(image, "{}".format(i + 1), (relative_x, relative_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    # 이미지 저장 코드
    # cv2.imwrite('./coordinate/meshed_img_' + str(num) + '.png', image)

cv2.imshow('dd', image)
cv2.waitKey(0)


# ================================================================================================================
def 눈():
    # ------------------------
    # 왼쪽 세로 평균
    eye_vt_left = (Eyeline_l_dict['161'][1] - Eyeline_l_dict['145'][1]) + (
                Eyeline_l_dict['160'][1] - Eyeline_l_dict['146'][1]) + \
                  (Eyeline_l_dict['159'][1] - Eyeline_l_dict['154'][1]) / 3

    # ------------------------
    # 왼쪽 가로
    eye_hr_left = Eyeline_l_dict['34'][0] - Eyeline_l_dict['156'][0]

    # ------------------------
    # 오른쪽 세로 평균
    eye_vt_right = (Eyeline_r_dict['385'][1] - Eyeline_r_dict['381'][1]) + (
                Eyeline_r_dict['386'][1] - Eyeline_r_dict['375'][1]) + \
                   (Eyeline_r_dict['387'][1] - Eyeline_r_dict['374'][1]) / 3

    # ------------------------
    # 세로 평균
    eye_avg_vt = (eye_vt_left + eye_vt_right) / 2

    # ------------------------
    # 오른쪽 가로
    eye_hr_right = Eyeline_r_dict['383'][0] - Eyeline_r_dict['264'][0]

    # -------------------------
    # 가로 평균
    eye_avg_hr = (eye_hr_left + eye_hr_right) / 2
    # ------------------------
    # 왼쪽 눈꼬리 기울기
    eye_grad_left = (Eyeline_l_dict['34'][1] - Eyeline_l_dict['156'][1]) / (
            Eyeline_l_dict['34'][0] - Eyeline_l_dict['156'][0])

    # ------------------------
    # 오른쪽 눈꼬리 기울기
    eye_grad_right = (Eyeline_r_dict['264'][1] - Eyeline_r_dict['383'][1]) / (
            Eyeline_r_dict['264'][0] - Eyeline_r_dict['383'][0])

    # ------------------------
    if eye_grad_left < 0:
        eye_gradient = -((abs(eye_grad_left) + eye_grad_right) / 2)
    else:
        eye_gradient = ((eye_grad_left + eye_grad_right) / 2)

    return eye_avg_vt, eye_avg_hr, eye_gradient


# ------------------------


print('=' * 60)
print(Eyeline_l_dict)
print(Eyeline_r_dict)
print(Face_area_dict)

# 세로길이 / 가로길이 / 눈꼬리 기울기
eyes = 눈()
print('세로길이 :', eyes[0], '가로길이 : ', eyes[1], '눈꼬리 기울기 : ', eyes[2])
