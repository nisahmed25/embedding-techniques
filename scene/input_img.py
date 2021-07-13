import numpy as np
import cv2


def inp_img():
    cap = cv2.VideoCapture('./../../../../vdo.mp4')
    success,img = cap.read()
    b,g,r = cv2.split(img)
    img = cv2.merge((r,g,b))
    pts = np.array([[10,150],[150,100],[300,150],[350,100],[310,20],[35,10]])
    return img, pts
    