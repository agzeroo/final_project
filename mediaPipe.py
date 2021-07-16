import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

path = 'C:/Users/user/Desktop/프로젝트/파이널프로젝트/face-mesh-generator-master/data/'   # 폴더 경로
os.chdir(path)   # 해당 폴더로 이동

# For static images:
IMAGE_FILES = []

files = os.listdir(path)
for file in files:
    if '.png' in file:
        f = cv2.imread(file)
        IMAGE_FILES.append(f)
    if '.jpg' in file:
        f = cv2.imread(file)
        IMAGE_FILES.append(f)
# print(IMAGE_FILES)

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
    cv2.imwrite(path + 'tmp/annotated_image_' + str(idx) + '.png', annotated_image)
    if not cv2.imwrite('tmp/annotated_image_' + str(idx) + '.png', annotated_image):
        raise Exception("Could not write image")