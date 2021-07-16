import cv2
import os

path = 'C:/Users/user/Desktop/프로젝트/파이널프로젝트/face-mesh-generator-master/data/'   # 폴더 경로
os.chdir(path)

IMAGE_FILES = [] # 이미지 리스트

files = os.listdir(path)
for file in files:
    if '.png' in file:
        f = cv2.imread(file)
        IMAGE_FILES.append(f)
    if '.jpg' in file:
        f = cv2.imread(file)
        IMAGE_FILES.append(f)
print(IMAGE_FILES)

for idx, image in enumerate(IMAGE_FILES):
    test_resize = cv2.resize(image, dsize=(256,256), interpolation=cv2.INTER_AREA)
    cv2.imwrite('./tmp/resized_image_' + str(idx) + '.png', test_resize)
    if not cv2.imwrite('./tmp/resized_image_' + str(idx) + '.png', test_resize):
        raise Exception("Could not write image")

'''
# 이미지 Crop
# Read the input image
os.getcwd()
img = cv2.imread('./data/222.jpg')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# Detect faces
faces = face_cascade.detectMultiScale(gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(256, 256))

# Draw rectangle around the faces and crop the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    faces = img[y:y + h, x:x + w]
    cv2.imshow("face", faces)
    cv2.imwrite('face.jpg', faces)

# Display the output
cv2.imwrite('detcted.jpg', img)
cv2.imshow('img', img)
cv2.waitKey()
'''

