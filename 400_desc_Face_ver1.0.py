import cv2
import mediapipe as mp
import os
import sympy
import numpy as np
import pandas as pd
import natsort

# 계산함수 ============================================================
# 1. 눈썹
def 눈썹() :
 
    # -------------------------------------------------
    # 긴 vs 짧음 - X 값 계산
    # ------------------------
    # 왼쪽 길이
    eyebrow_hr_subtraction_left = eyebrow_left_top_dict['108'][0] - eyebrow_left_top_dict['71'][0]

    # ------------------------
    # 오른쪽 길이
    eyebrow_hr_subtraction_right = eyebrow_right_top_dict['301'][0] - eyebrow_right_top_dict['337'][0]


    # ------------------------
    # 왼쪽, 오른쪽 길이의 평균
    avg_eb_sub = (eyebrow_hr_subtraction_left + eyebrow_hr_subtraction_right) / 2



    # -------------------------------------------------
    # ------------------------
    # 굵음 vs 얇음 - Y 값 계산
    # 왼쪽 굵기
    eyebrow_vt_subtraction_left_left = eyebrow_left_top_dict['64'][1] - eyebrow_left_btm_dict['54'][1]
    eyebrow_vt_subtraction_left_middle = eyebrow_left_top_dict['106'][1] - eyebrow_left_btm_dict['53'][1]
    eyebrow_vt_subtraction_left_right = eyebrow_left_top_dict['67'][1] - eyebrow_left_btm_dict['66'][1]

    avg_vt_left = (eyebrow_vt_subtraction_left_left + eyebrow_vt_subtraction_left_middle + eyebrow_vt_subtraction_left_right) / 3
    avg_vt_left = abs(avg_vt_left)


    # ------------------------
    # 오른쪽 굵기
    eyebrow_vt_subtraction_right_left = eyebrow_right_top_dict['297'][1] - eyebrow_right_btm_dict['296'][1]
    eyebrow_vt_subtraction_right_middle = eyebrow_right_top_dict['335'][1] - eyebrow_right_btm_dict['283'][1]
    eyebrow_vt_subtraction_right_right = eyebrow_right_top_dict['294'][1] - eyebrow_right_btm_dict['284'][1]

    avg_vt_right = (eyebrow_vt_subtraction_right_left + eyebrow_vt_subtraction_right_middle + eyebrow_vt_subtraction_right_right) / 3
    avg_vt_right = abs(avg_vt_right)


    # ------------------------
    # 왼쪽, 오른쪽의 평균 굵기
    avg_vt = (avg_vt_left + avg_vt_right) / 2


    # =============================================================================================================================
    # 올라감 vs 내려감

    # ------------------------
    # 왼쪽 눈썹 기울기( 마이너스 값이므로, 절대값 해준다. )
    eyebrow_gradient_left = (eyebrow_left_top_dict['71'][1] - eyebrow_left_btm_dict['56'][1]) / (eyebrow_left_top_dict['71'][0] - eyebrow_left_btm_dict['56'][0])
    eyebrow_gradient_left = abs(eyebrow_gradient_left)

    # ------------------------
    # 오른쪽 눈썹 기울기
    eyebrow_gradient_right = (eyebrow_right_top_dict['301'][1] - eyebrow_right_btm_dict['286'][1]) / (eyebrow_right_top_dict['301'][0] - eyebrow_right_btm_dict['286'][0])


    # ------------------------
    # 기울기 높은 것으로 선택.
    if eyebrow_gradient_left >= eyebrow_gradient_right :
        eyebrow_gradeint = eyebrow_gradient_left
    else :
        eyebrow_gradeint = eyebrow_gradient_right


    return avg_eb_sub, avg_vt, eyebrow_gradeint

# -------------------------------------------------------------------
# 2. 눈
def 눈():
    # ------------------------
    # 왼쪽 세로 평균
    eye_vt_left = (Eyeline_l_dict['145'][1] - Eyeline_l_dict['161'][1]) + (
            Eyeline_l_dict['146'][1] - Eyeline_l_dict['160'][1]) + \
                    (Eyeline_l_dict['154'][1] - Eyeline_l_dict['159'][1]) / 3

    # ------------------------
    # 왼쪽 가로
    eye_hr_left = Eyeline_l_dict['156'][0] - Eyeline_l_dict['34'][0]

    # ------------------------
    # 오른쪽 세로 평균
    eye_vt_right = (Eyeline_r_dict['381'][1] - Eyeline_r_dict['385'][1]) + (
            Eyeline_r_dict['375'][1] - Eyeline_r_dict['386'][1]) + \
                    (Eyeline_r_dict['374'][1] - Eyeline_r_dict['387'][1]) / 3
    # ------------------------
    # 오른쪽 가로
    eye_hr_right = Eyeline_r_dict['264'][0] - Eyeline_r_dict['383'][0]

    # ------------------------
    # 세로 평균
    eye_avg_vt = (eye_vt_left + eye_vt_right) / 2

    # -------------------------
    # 가로 평균
    eye_avg_hr = (eye_hr_left + eye_hr_right) / 2

    #--------------------------
    # 비율 - 세로/가로
    eye_ratio = eye_avg_vt/eye_avg_hr

    # ------------------------
    # 왼쪽 눈꼬리 기울기
    eye_grad_left = (Eyeline_l_dict['34'][1] - Eyeline_l_dict['156'][1]) / (
            Eyeline_l_dict['34'][0] - Eyeline_l_dict['156'][0])

    # ------------------------
    # 오른쪽 눈꼬리 기울기
    eye_grad_right = (Eyeline_r_dict['264'][1] - Eyeline_r_dict['383'][1]) / (
            Eyeline_r_dict['264'][0] - Eyeline_r_dict['383'][0])

    # ------------------------
    # 얼굴 좌우 비율

    face_hr_distance = Face_area_dict['448'][0] - Face_area_dict['235'][0]
    face_vt_distance = Face_area_dict['153'][1] - Face_area_dict['11'][1]
    face_ratio = face_hr_distance/face_vt_distance

    if eye_grad_left < 0:
        eye_gradient = -((abs(eye_grad_left) + eye_grad_right) / 2)
    else:
        eye_gradient = ((eye_grad_left + eye_grad_right) / 2)

    return eye_avg_vt, eye_avg_hr, eye_ratio, eye_gradient, face_ratio

# -------------------------------------------------------------------
# 3. 코
def nose():
    
    # ------------------------
    # 코 길이
    nose_length = nose_length_dict['95'][1]-nose_length_dict['169'][1]

    # ------------------------
    # 코 폭
    nose_width = R_width_dict['421'][0] - \
        L_width_dict['199'][0]


    # ------------------------
    # 콧볼 폭

    nostril = R_nostril_dict['279'][0] - \
        L_nostril_dict['103'][0]


    return nose_length, nose_width, nostril

# -------------------------------------------------------------------
# 4. 입
def Mouth() :
    
    #============================================================
    # 입크기
    
    lip_width = round(abs(((upper_lips_dict['410'][0] - upper_lips_dict['62'][0])/2) * ((upper_lips_dict['38'][1] - lower_lips_dict['18'][1])/2) * 3.14), 2)


    #=============================================================
    # 윗입술이 두꺼운사람 vs 얇은 사람

    upperlipThickness_left = upper_lips_dict['38'][1] - upper_lips_dict['83'][1]
    upperlipThickness_middle = upper_lips_dict['1'][1] - upper_lips_dict['14'][1]
    upperlipThickness_right = upper_lips_dict['268'][1] - upper_lips_dict['269'][1]
    
    upperlipThickness = round(abs((upperlipThickness_left + upperlipThickness_middle + upperlipThickness_right)/3),2)




    #=====================================================================================
    # 아랫입술이 두꺼운사람 vs 얇은 사람

    lowerlipThickness_left = lower_lips_dict['88'][1] - lower_lips_dict['85'][1]
    lowerlipThickness_middle = lower_lips_dict['179'][1] - lower_lips_dict['182'][1]
    lowerlipThickness_right = lower_lips_dict['318'][1] - lower_lips_dict['315'][1]

    lowerlipThickness = round(abs(( lowerlipThickness_left + lowerlipThickness_middle + lowerlipThickness_right)/3),2)


    
    #========================================================================================

    #입꼬리가 올라간 사람 vs 내려간 사람
    # 양의 기울기 = 입꼬리 올라감
    # 음의 기울기 = 입꼬리 내려감
    lip_tail = round((upper_lips_dict['14'][1] - upper_lips_dict['410'][1] ) / (upper_lips_dict['410'][0] - upper_lips_dict['14'][0]),2)


    return lip_width, upperlipThickness, lowerlipThickness, lip_tail

# -------------------------------------------------------------------
# 5. 얼굴형

def lsa(x, y, n=5):
    """ (n-1)차 다항식 최소제곱근사 구현하는 함수, 입력 안하면 기본값 4차 다항식
        input1 : x 값들 담은 리스트
        input2 : y 값들 담은 리스트
        output : y = a_0 + a_1 x + a_2 x^2 .. 에서 [a_0, a_1, a_2, ..] 값들을 ndarray 형태로 반환
    """
    temp1, temp2 = [], []

    for x_val in x:
        temp = []

        for i in range(n):
            temp.append(x_val ** i)

        temp1.append(temp)

    for y_val in y:
        temp2.append([y_val])

    A = np.array(temp1)
    b = np.array(temp2)

    return np.linalg.inv(A.T @ A) @ A.T @ b


def 얼굴형(x, y):
    """ x, y 입력받아 근사다항식 만들고, 특정점에서의 미분, 적분값들 이용해서 얼굴형 분류하는 함수 """

    # 1. x, y 값 설정
    # 1.1 (중심값(153번째 포인트) 를 (0, 0)) 으로 만들기
    # 1.2 y값 음수로 만들기 (아래로 볼록하게 만들기)
    temp_x, temp_y = x[5], y[5]

    for i in range(len(x)):
        x[i] -= temp_x
        y[i] -= temp_y
        y[i] = -y[i]

    # 2. 최소제곱근사
    a1 = lsa(x, y)  # 주의, 다항식 차수 맞춰야
    t = sympy.symbols('t')

    # 3. 다항식 만들기
    equ = a1[0][0]

    for i in range(1, len(a1)):
        equ += a1[i][0] * (t ** i)

    # 4. 미분값 구하기
    diff1 = sympy.diff(equ)

    d1 = (diff1.subs(t, x[-1]) - diff1.subs(t, x[0])) / 2
    d2 = (diff1.subs(t, x[6]) - diff1.subs(t, x[4])) / 2

    # 5. 적분값 구하기
    area = sympy.integrate(equ, (t, x[0], x[-1])) * 10

    return d1, d2, area


# =====================================================================================
# 부위별 포인트
# -------------------------------------------------------------------------
# 1. 눈썹 포인트
eyebrow_points = [71, 64, 106, 67, 108, 47, 54, 53, 66, 56, 337, 297, 335, 294, 301, 286, 296, 283, 284, 277]

# 왼쪽 눈썹 포인트 위, 아래
eyebrow_left_top = [71, 64, 106, 67, 108]
eyebrow_left_btm = [47, 54, 53, 66, 56]

# 오른쪽 눈썹 포인트 위, 아래
eyebrow_right_top = [337, 297, 335, 294, 301]
eyebrow_right_btm = [286, 296, 283, 284, 277]

# 눈썹 계산값 
eyebrow_desc_dict = {}

# -----------------------------------------------------
# 눈썹 - 숫자값(key)에 좌표를 담아주기 위한 dict 생성
# 왼쪽 위
eyebrow_left_top_dict = {}
# 왼쪽 아래
eyebrow_left_btm_dict = {}


# 오른쪽 위
eyebrow_right_top_dict = {}
# 오른쪽 아래
eyebrow_right_btm_dict = {}
# -------------------------------


# -------------------------------------------------------------------------
# 2. 눈 포인트
# 눈 포인트
eye_points = [34, 161, 145, 160, 146, 159, 154, 156, 383, 385, 381, 386, 375, 387, 374, 264, 235, 448, 11, 153]

# 얼굴면적(가로좌, 가로우, 세로위, 세로아래)
Face_area = [235, 448, 11, 153]

# 왼쪽 눈썹 포인트 좌, 위, 아래, 우
Eyeline_l = [34, 161, 145, 160, 146, 159, 154, 156]

# 오른쪽 눈썹 포인트 좌, 위, 아래, 우
Eyeline_r = [383, 385, 381, 386, 375, 387, 374, 264]

# -------------------------------
# 눈
# 왼쪽
Eyeline_l_dict = {}

# 오른쪽
Eyeline_r_dict = {}

# 얼굴면적용(가로좌, 가로우, 세로위, 세로아래)
Face_area_dict = {}

# 눈 계산값 
eye_desc_dict = {}


# ---------------------------------------------
# 3. 코 포인트
nose_points = [169, 7, 198, 196, 6, 5, 2, 20, 95, 123, 197, 4, 237, 199, 132, 50, 103, 130, 65, 99, 241, 100,
               352, 420, 249, 282, 364, 361, 280, 332, 359, 259, 456, 306, 291]

# 코 길이
nose_length_points = [169, 7, 198, 196, 6, 5, 2, 20, 95]
# 콧대 폭
R_width = [352, 420, 457, 421]
L_width = [123, 197, 237, 199]

# 콧볼
R_nostril = [361, 280, 279, 295, 456, 461]

L_nostril = [132, 50, 103, 65, 99, 241]

# 코
# 코 길이
nose_length_dict = {}

# 콧대 폭
# 오른쪽
R_width_dict = {}
# 왼쪽
L_width_dict = {}

# 콧볼
# 오른쪽
R_nostril_dict = {}

# 왼쪽
L_nostril_dict = {}


# 코값 계산값
nose_desc_dict = {}


# ---------------------------------------------
# 4. 입 포인트
mouth_points = [62, 38, 1, 14, 83, 268, 269, 410, 179, 88, 182, 85, 18, 315, 318]


# 윗입술, 아랫입술
upper_lips = [62, 38, 1, 14, 83, 268, 269, 410]
lower_lips = [179, 88, 182, 85, 18, 315, 318]

upper_lips_dict = {}
lower_lips_dict = {}

# 입 계산값 
mouth_desc_dict = {}

# ---------------------------------------------
# 5. 얼굴형 포인트

FaceLine_points = [216, 173, 137, 151, 177, 153, 379, 395, 398, 436, 402]

# 1인 얼굴형 계산값
faceLine_dict = {}

# 얼굴형 계산값
faceLine_desc_dict = {}


# =====================================================================================
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


path = 'C:/Users/user/Desktop/faceimg/Filtered_img'   # 폴더 경로
os.chdir(path)   # 해당 폴더로 이동

# For static images:
IMAGE_FILES = []
files_tmp = os.listdir(path)
# print(files_tmp)
files = natsort.natsorted(files_tmp)
# print(files)

for file in files:
    if '.png' in file:
        f = cv2.imread(file)
        # 이미지 256x256 resizeㄹ
        # f = cv2.resize(f, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        IMAGE_FILES.append(f)
    if '.jpg' in file:
        f = cv2.imread(file)
        # f = cv2.resize(f, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        IMAGE_FILES.append(f)


# # 진행률
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


#---------------------- 탭 두번 선 --------------------------------------------------------


        # 얼굴 랜드마크 468개 순서체크 이미지 생성
        # print(results.multi_face_landmarks[0].landmark[0].y)


        # ----------------------------------------------------------------------------------
        # for 문
        num = 1


        for face in results.multi_face_landmarks:
            for (i, landmark) in enumerate(face.landmark):
                x = landmark.x
                y = landmark.y
                z = landmark.z
                # print(f'{num}번째 랜드마크', x, y, z)

                shape = image.shape
                relative_x = int(x * shape[1])
                relative_y = int(y * shape[0])

                # print(relative_x, relative_y)

                cv2.circle(image, (relative_x, relative_y),
                        radius=1, color=(0, 255, 0), thickness=1)


                # -------------------------------
                # 1. 눈썹
                if num in eyebrow_left_top :
                    eyebrow_left_top_dict[f'{num}'] = relative_x, relative_y, x, y, z
                    
                
                elif num in eyebrow_left_btm :
                    eyebrow_left_btm_dict[f'{num}'] = relative_x, relative_y, x, y, z
                    

                elif num in eyebrow_right_top :
                    eyebrow_right_top_dict[f'{num}'] = relative_x, relative_y, x, y, z

                elif num in eyebrow_right_btm :
                    eyebrow_right_btm_dict[f'{num}'] = relative_x, relative_y, x, y, z

                # -------------------------------
                # 2. 눈
                if num in Eyeline_l:
                    Eyeline_l_dict[f'{num}'] = relative_x, relative_y, x, y, z

                elif num in Eyeline_r:
                    Eyeline_r_dict[f'{num}'] = relative_x, relative_y, x, y, z

                elif num in Face_area:
                    Face_area_dict[f'{num}'] = relative_x, relative_y, x, y, z

                # -------------------------------
                # 3. 코                
                if num in nose_length_points:
                    nose_length_dict[f'{num}'] = relative_x, relative_y

                elif num in R_width:
                    R_width_dict[f'{num}'] = relative_x, relative_y

                elif num in L_width:
                    L_width_dict[f'{num}'] = relative_x, relative_y

                elif num in R_nostril:
                    R_nostril_dict[f'{num}'] = relative_x, relative_y

                elif num in L_nostril:
                    L_nostril_dict[f'{num}'] = relative_x, relative_y


                # -------------------------------
                # 4. 입
                if num in upper_lips :
                    upper_lips_dict[f'{num}'] = relative_x, relative_y, x, y, z

                elif num in lower_lips:
                    lower_lips_dict[f'{num}'] = relative_x, relative_y, x, y, z


                # -------------------------------
                # 5. 얼굴형태
                if num in FaceLine_points:
                    faceLine_dict[f'{num}'] = x, y, z

                # -------------------------------

                num += 1

                cv2.putText(image, "{}".format(i + 1), (relative_x, relative_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            # cv2.imwrite(f'./tmp/annotated_image_{str(idx)}.png', image)
            # if not cv2.imwrite(f'./tmp/annotated_image_{str(idx)}.png', image):
            #     raise Exception("Could not write image")

        # =====================================================================================================================================================
        # temp_ 사용하는 이유 - 400장으로 통계값 내기 위해서!
        # 만약 한 장 이라면? - 한 장에 대한 통계값을 계산할 수 있음!
        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        # 1. 눈썹
        temp_eyebrow_desc_key = f'{idx}'
        temp_eyebrow_desc = 눈썹()
        eyebrow_desc_dict[f'{temp_eyebrow_desc_key}'] = temp_eyebrow_desc


        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        # 2. 눈
        temp_eye_desc_key = f'{idx}'
        temp_eye_desc = 눈()
        eye_desc_dict[f'{temp_eye_desc_key}'] = temp_eye_desc


        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        # 3. 코
        temp_nose_desc_key = f'{idx}'
        temp_nose_desc = nose()
        nose_desc_dict[f'{temp_nose_desc_key}'] = temp_nose_desc


        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        # 4. 입
        temp_mouth_desc_key = f'{idx}'
        temp_mouth_desc = Mouth()
        mouth_desc_dict[f'{temp_mouth_desc_key}'] = temp_mouth_desc


        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        # 5. 얼굴형
        faceLine_x = []
        faceLine_y = []
        for i in range(len(FaceLine_points)):
            faceLine_x.append(faceLine_dict[f'{FaceLine_points[i]}'][0])
            faceLine_y.append(faceLine_dict[f'{FaceLine_points[i]}'][1])

        faceLine_x, faceLine_y = np.array(faceLine_x), np.array(faceLine_y)

        temp_faceLine_desc_key = f'{idx}'
        temp_faceLine_desc = 얼굴형(faceLine_x, faceLine_y)
        faceLine_desc_dict[f'{temp_faceLine_desc_key}'] = temp_faceLine_desc

        # =====================================================================================================================================================



#---------------------- 탭 두번 선 --------------------------------------------------------


# ------------------------------------------------
# 1. 눈썹
df_eyebrow_desc = pd.DataFrame(eyebrow_desc_dict)
df_eyebrow_desc = df_eyebrow_desc.T
df_eyebrow_desc.columns = ['눈썹_길이', '눈썹_굵기', '눈썹_기울기']


# ------------------------------------------------
# 2. 눈
df_eye_desc = pd.DataFrame(eye_desc_dict)
df_eye_desc = df_eye_desc.T
df_eye_desc.columns = ['눈_세로길이', '눈_가로길이', '비율', '눈_눈꼬리기울기', '얼굴비율']


# ------------------------------------------------
# 3. 코
df_nose_desc = pd.DataFrame(nose_desc_dict)
df_nose_desc = df_nose_desc.T
df_nose_desc.columns = ['코_길이', '코_폭', '콧볼_폭']


# ------------------------------------------------
# 4. 입
df_mouth_desc = pd.DataFrame(mouth_desc_dict)
df_mouth_desc = df_mouth_desc.T
df_mouth_desc.columns = ['입_크기', '입_윗입술(두께)', '입_아랫입술(두께)', '입_입꼬리']

# ------------------------------------------------
# 4. 얼굴형
df_faceLine_desc = pd.DataFrame(faceLine_desc_dict)
df_faceLine_desc = df_faceLine_desc.T
df_faceLine_desc.columns = ['턱기울기1', '턱기울기2', '턱아래면적']

# ==============================================================================
# 사진별 데이터를 데이터프레임으로 취합
df_all = pd.DataFrame()
df_list = [df_eyebrow_desc, df_eye_desc, df_nose_desc, df_mouth_desc, df_faceLine_desc]

for face_parts_df in df_list:
    df_all = pd.concat([df_all, face_parts_df], axis=1)


# CSV파일 저장
df_all.to_csv('Images_values.csv', encoding='utf-8-sig')
# print('csv파일 저장 완료')




# # ====================================================================================================================================================
# 최종 desc = {}
# 분류할 때 사용할 예정
desc = {}

# 1. 눈썹 부위 - 첫번째 : 길이 평균 // 두번째 : 굵기 평균 // 세번째 : 기울기 평균
desc['eyebrow'] =  (df_all['눈썹_길이'].mean(), 
                   df_all['눈썹_굵기'].mean(), 
                   df_all['눈썹_기울기'].mean())

desc['eye'] =  (df_all['눈_세로길이'].mean(), 
                   df_all['눈_가로길이'].mean(), 
                   df_all['비율'].mean(),
                   df_all['눈_눈꼬리기울기'].mean(),
                   df_all['얼굴비율'].mean())
                   
desc['nose'] =  (df_all['코_길이'].mean(), 
                   df_all['코_폭'].mean(), 
                   df_all['콧볼_폭'].mean())

desc['mouth'] =  (df_all['입_크기'].mean(), 
                   df_all['입_윗입술(두께)'].mean(), 
                   df_all['입_아랫입술(두께)'].mean(),
                   df_all['입_입꼬리'].mean())


desc['face'] =  (df_all['턱기울기1'].mean(),
                   df_all['턱기울기2'].mean(),
                   df_all['턱아래면적'].mean())


# # 통합 desc 확인
# print('-'*60)
print('*****이미지 분석 완료*****')
# print(desc)




