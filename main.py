import base64
import io
import math
import os
import pathlib
import sys
from os.path import exists
import eventlet
import socketio
from PIL import Image
import cv2
import numpy as np
import time
import socket
import json
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# 폐지된 기능에 대한 경고를 무시

sio = socketio.Server()
app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})

# environ : http 헤더를 포함한 http 리퀘스트 데이터를 담는 WSGI 표준 딕셔너리
#클라가 연결할 때 event를 보냄
@sio.event
def connect(sid, environ):
    print('connect ', sid)

# now : 원본이미지 저장시간
# human_list : 선택된 인덱스 리스트
# pixelSize : 모자이크 픽셀 크기 (기본 20)
@sio.event()
def getBlurImg(sid, now, human_list, pixel_size=20):
    # Json 문자열을 Python 객체로 변환
    human_list = json.loads(human_list)
    human = []

    # 원본 이미지를 불러옴
    natImage = Image.open(now + '/natural.jpg')
    # BGR 사진을 RGB 사진으로 변환
    frame = cv2.cvtColor(np.array(natImage), cv2.COLOR_BGR2RGB)

    # 선택된 사람이 없다면 원본 이미지 보내기
    if not human_list:
        bt = io.BytesIO()
        natImage.save(bt, format="JPG")
        im_bytes = bt.getvalue()
        nat64 = base64.b64encode(im_bytes)
        sio.emit('get_blur', nat64)
        print('index not selected')
    # 인덱스 값이 있다면 해당 얼굴 모자이크 처리 후 반환
    else:
        for i in human_list:
            rate = int((human[i][2] - human[i][0] + human[i][3] - human[i][1]) / pixel_size)
            # 탐지된 객체의 크기의 가로 길이와 세로 길이를 합한 값을 사용해서 비율을 구함
            face_img = frame[human[i][1]:human[i][3], human[i][0]:human[i][2]]  # 탐지된 얼굴 이미지 crop
            face_img = cv2.resize(face_img, ((human[i][2] - human[i][0]) // rate, (human[i][3] - human[i][1]) // rate))  # rate 만큼 축소
            face_img = cv2.resize(face_img, (human[i][2] - human[i][0], human[i][3] - human[i][1]), interpolation=cv2.INTER_AREA)  # 확대
            frame[human[i][1]:human[i][3], human[i][0]:human[i][2]] = face_img  # 탐지된 얼굴 영역 모자이크 처리
        # Numpy 배열로 되어있는 이미지 배열을 PIL 이미지로 변환
        result_blue = Image.fromarray(frame)
        # BGR 사진을 RGB 사진으로 변환
        result = cv2.cvtColor(np.array(result_blue), cv2.COLOR_BGR2RGB)
        # Numpy 배열로 되어있는 이미지 배열을 PIL 이미지로 변환
        result_img = Image.fromarray(result)
        bt = io.BytesIO()
        result_img.save(bt, format="PNG")
        im_bytes = bt.getvalue()
        res64 = base64.b64encode(im_bytes).decode('utf-8')
        sio.emit('get_blur', {"image" : res64})
        result_img.show()
        print('blur completed')

# now : 원본 이미지 저장 시간
# image : 원본 이미지
# reliability : 신뢰도

@sio.event()
def detectFace(sid, now, image=None, reliability=0.3):
    # 객체 인식 모델
    model = 'opencv_face_detector_uint8.pb'
    # 모델 라벨
    config = 'opencv_face_detector.pbtxt'

    face_list = []
    #readNet() 함수는 model과 config 파일 이름 확장자를 분석하여
    # 내부에서 해당 프레임워크에 맞는 readNetFromXXX() 형태의 함수를 다시 호출
    # 여기선 .pb -> readNetFromTensorflow()
    net = cv2.dnn.readNet(model, config)

    # 클라이언트에서 처음 이미지를 보냈을 때만
    # 현재 시간 폴더 생성 후 원본 이미지 저장
    if image is not None:
        # 원본 이미지 저장 시간을 이름으로 가지는 폴더 생성
        os.mkdir(now)
        # base64 to image
        imgdata = base64.b64decode(image)
        dataBytesIO = io.BytesIO(imgdata)
        cvtImg = Image.open(dataBytesIO)
        naturalImg = cv2.cvtColor(np.array(cvtImg), cv2.COLOR_BGR2RGB)

        #now 변수를 이름으로 하는 폴더에 원본이미지 natural.jpg 라는 이름으로 저장
        cv2.imwrite(now + '/natural.jpg', np.array(naturalImg))

    # 원본 이미지를 불러옴
    natImage = Image.open(now + '/natural.jpg')
    frame = cv2.cvtColor(np.array(natImage), cv2.COLOR_BGR2RGB)

    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detect = net.forward()

    detect = detect[0, 0, :, :]
    (h, w) = frame.shape[:2]

    for i in range(detect.shape[0]):
        confidence = detect[i, 2]
        if confidence < reliability:
            continue

        x1 = int(detect[i, 3] * w)
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)
        face_img = frame[y1:y2, x1:x2]
        face_list.append(face_img)

    # 다음 코드를 실행하여 사용자의 얼굴 리스트를 클라이언트에게 전송
    face_counter = 0
    face64list = []
    for face_array in face_list:
        # 얼굴 리스트 저장
        natFace = cv2.cvtColor(face_array, cv2.COLOR_BGR2RGB)  # 사진 색을 원본으로 되돌림
        face = Image.fromarray(natFace)  # PIL로  array를 이미지로 변환 type=PIL.Image.Image
        face.save(now + '/' + str(face_counter) + '.jpg')
        print('save finish')
        with open(now + '/' + str(face_counter) + '.jpg', 'rb') as img:
            face64 = base64.b64encode(img.read()).decode('utf-8')
        face64list.append(face64)
        face_counter += 1
    print(face_counter)
    sio.emit('get_faces', {"image": face64list})


@sio.event
def disconnect(sid):
    print('disconnect ', sid)

# if __name__ == '__main__':
#     eventlet.wsgi.server(eventlet.listen(("ip address", port num)), app)
