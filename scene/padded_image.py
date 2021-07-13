import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import image
from tensorflow.image import ResizeMethod

def crop_pad(image_size, img, pts):  
   
    # crop and add bg 
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    bg = np.ones_like(croped, np.uint8)*0
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg+ dst
    target_height, target_width = image_size
    padded_img = image.resize_with_pad(
        dst2, target_height, target_width, method=ResizeMethod.NEAREST_NEIGHBOR,
        antialias=False)
    padded_img = np.expand_dims(padded_img, axis=0)
    return padded_img