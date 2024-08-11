import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# STEP 2: create inference instance (추론기 생성)
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

# STEP 3: load input image (추론 데이터 가져오기)
img1 = cv2.imread("img\iu1.jpg")
img2 = cv2.imread("img\iu2.jpg")

# STEP 4: inference (추론)
faces1 = app.get(img1)
faces2 = app.get(img2)
print(len(faces1))
print(len(faces2))

# STEP 5: draw detection result
# assert len(faces1)==1
# assert len(faces2)==1
# # rimg = app.draw_on(img, faces)
# # cv2.imwrite("res\output.jpg", rimg)

# STEP 5: face similarity
# then print all-to-all face similarity (얼굴 간 유사도 측정)
feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32) # 한번에 행렬연산하기 위함
feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)

sims = np.dot(feat1, feat2.T)
print(sims) # 0.4 기준으로 동일인물인지 판별
