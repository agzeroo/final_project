import cv2
import mediapipe as mp
import os
from tqdm import tqdm
from time import sleep

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

path = 'C:/final_project/final_project/data/'   # 폴더 경로
os.chdir(path)   # 해당 폴더로 이동

# For static images:
IMAGE_FILES = []

files = os.listdir(path)
for file in files:
    if '.png' in file:
        f = cv2.imread(file)
        # 이미지 256x256 resize
        f = cv2.resize(f, dsize=(1500, 1500), interpolation=cv2.INTER_AREA)
        IMAGE_FILES.append(f)
    if '.jpg' in file:
        f = cv2.imread(file)
        f = cv2.resize(f, dsize=(1500, 1500), interpolation=cv2.INTER_AREA)
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
        cv2.imwrite('./tmp/annotated_image_' +
                    str(idx) + '.png', annotated_image)
        if not cv2.imwrite('./tmp/annotated_image_' + str(idx) + '.png', annotated_image):
            raise Exception("Could not write image")


# 얼굴 랜드마크 468개 순서체크 이미지 생성
# print(results.multi_face_landmarks[0].landmark[0].y)
num = 1
for face in results.multi_face_landmarks:
    for (i, landmark) in enumerate(face.landmark):
        x = landmark.x
        y = landmark.y
        z = landmark.z
        print(f'{num}번째 랜드마크', x, y, z)

        shape = image.shape
        relative_x = int(x * shape[1])
        relative_y = int(y * shape[0])

        cv2.circle(image, (relative_x, relative_y),
                   radius=1, color=(0, 255, 0), thickness=1)
        cv2.imwrite('./coordinate/meshed_img_' + str(num) + '.png', image)
        num += 1

        cv2.putText(image, "{}".format(i + 1), (relative_x, relative_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)


cv2.imshow('dd', image)
cv2.waitKey(0)
