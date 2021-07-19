import numpy as np
import dlib
import cv2

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))
NOSE = list(range(27, 36))
EYEBROWS = list(range(17, 27))
JAWLINE = list(range(1, 17))
ALL = list(range(0, 68))
EYES = list(range(36, 48))

# 학습된 모델
predictor_file = 'C:/Users/master15/Desktop/openCV_dnn_v2//model/shape_predictor_68_face_landmarks.dat'
# 이미지 파일
image_file = 'C:/Users/master15/Desktop/openCV_dnn_v2/image/marathon_03.jpg'

# get_frontal_face_detector(): 정면 얼굴
detector = dlib.get_frontal_face_detector()

# 68개의 점으로 하는 파일 가져오기
predictor = dlib.shape_predictor(predictor_file)

# imread : 이미지파일 cv로 읽어오기
image = cv2.imread(image_file)

# 노이즈 제거를 위해 Gray색상으로 바꿔서 진행, 인식률 높이기 위함
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# gray 이미지로 detector
# 1: detection할 때 이미지 레이어 몇번 적용할지
# 보통 1하면 왠만한 큰사진 다 됨
rects = detector(gray, 1)

# 몇명의 얼굴이 나오는지 print
print("Number of faces detected: {}".format(len(rects)))

# rects : 배열
# print(rects) 결과 : rectangles[[(241,118)(464,341)]]
for (i, rect) in enumerate(rects):
    points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
    # ALL : 얼굴전체의 점을 보여줌.
    # 눈, 코 등 입력하면 원하는 부분의 점만 출력
    show_parts = points[ALL]
    for (i, point) in enumerate(show_parts):
        x = point[0, 0]  # x좌표
        y = point[0, 1]  # y좌표

        # (0, 255, 255) : 노란색
        # -1 : 다채워진 점
        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
        # 점에 번호를 써준다
        cv2.putText(image, "{}".format(i + 1), (x, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

cv2.imshow("Face Landmark", image)
cv2.waitKey(0)
