import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import copy


def zad1():
    img = cv2.imread('img.png', 0)

    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, img, color=(255, 0, 0))

    print("Keypoints: ", len(kp))

    cv2.imshow("name", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def zad2():
    img = cv2.imread('1.png', 0)
    rows, cols = img.shape

    orb = cv2.ORB_create()
    kp1 = orb.detect(img, None)

    kp1, des1 = orb.compute(img, kp1)

    imgo = cv2.drawKeypoints(img, kp1, img, color=(0, 255, 0), flags=0)


    M = np.float32([[1, 0, 100], [0, 1, 50]])
    img2 = cv2.warpAffine(img, M, (cols, rows))

    kp2 = orb.detect(img2, None)

    kp2, des2 = orb.compute(img2, kp2)

    img2o = cv2.drawKeypoints(img2, kp2, img2, color=(0, 255, 0), flags=0)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img, kp1, img2, kp2, matches[:10], img2, flags=2)

    cv2.imshow('name1', img3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def zad3():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    img = cv2.imread('twarze.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


zad3()
