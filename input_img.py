"""
Input Image
"""
import numpy as np
import cv2

# pylint: disable=bad-indentation

def inp_img():
  """
  Capture Image
  """
  cap = cv2.VideoCapture('./../../../../vdo.mp4')
  _, img = cap.read()
  _b, _g, _r = cv2.split(img)
  img = cv2.merge((_r, _g, _b))
  pts = np.array([[10, 150], [150, 100], [300, 150], [350, 100], [310, 20], [35, 10]])
  return img, pts
