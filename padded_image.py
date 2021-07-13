"""
Padded Image
"""
import cv2
import numpy as np
from tensorflow import image

# pylint: disable=bad-indentation

def crop_pad(image_size, img, pts):
  """
  crop and add bg
  """
  rect = cv2.boundingRect(pts)
  _x, _y, _w, _h = rect
  croped = img[_y:_y + _h, _x:_x + _w].copy()
  pts = pts - pts.min(axis=0)
  mask = np.zeros(croped.shape[:2], np.uint8)
  cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
  dst = cv2.bitwise_and(croped, croped, mask=mask)
  _bg = np.ones_like(croped, np.uint8) * 0
  cv2.bitwise_not(_bg, _bg, mask=mask)
  dst2 = _bg + dst
  target_height, target_width = image_size
  padded_img = image.resize_with_pad(dst2,
                                     target_height,
                                     target_width,
                                     method=image.ResizeMethod.NEAREST_NEIGHBOR,
                                     antialias=False)
  padded_img = np.expand_dims(padded_img, axis=0)
  return padded_img
