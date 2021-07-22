import cv2
import os

# C:\Users\user\Desktop\faceimg\Middle_Resolution\19062421\S001\L1\E01\C7.jpg 이미지 예시
# 346 x 234 원본사이즈
path = 'c:/Users/user/Desktop/faceimg/Middle_Resolution/' # 폴더 경로

# For static images:
folders = os.listdir(path)

for idx, folder in enumerate(folders):
  folder_path = f'C:/Users/user/Desktop/faceimg/Middle_Resolution/{folders[idx]}/S001/L1/E01/'
  files = os.listdir(folder_path)
  for file in files:
    if 'C7.jpg' in file:
      file = os.path.join(folder_path, file)
      img = cv2.imread(file)
      img = cv2.resize(img, dsize=(1960, 1325), interpolation=cv2.INTER_AREA)
      cv2.imwrite('c:/Users/user/Desktop/faceimg/Filtered_img/img_' + str(idx) + '.png', img)
      if not cv2.imwrite('c:/Users/user/Desktop/faceimg/Filtered_img/img_' + str(idx) + '.png', img):
          raise Exception("Could not write image")
print('이미지 분류 완료!')