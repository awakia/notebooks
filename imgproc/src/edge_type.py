import math
import cv2
import numpy as np

import visualize

# https://en.wikipedia.org/wiki/Sobel_operator
def calc(gray):
  gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
  # visualize.img(gx, "seismic")
  gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
  # visualize.img(gy, "seismic")
  g = np.sqrt(gx * gx + gy * gy)
  gmax = np.max(g)
  theta = np.arctan2(gy, gx)
  print("gmax:", gmax)
  brightness = (g * (255 / gmax)).astype(np.uint8)
  # hue = ((theta + math.pi) * (180 / (2 * math.pi)) % 180).astype(np.uint8)
  hue = ((theta + math.pi) * (360 / (2 * math.pi)) % 180).astype(np.uint8)
  # saturation = np.full(gray.shape, 255, dtype=np.uint8)
  # hsv = np.dstack((hue, saturation, brightness))
  # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
  # visualize.img(bgr)
  return hue, brightness

FILTERS = (
  np.array([
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,9,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
  ], dtype=np.uint8),
  np.array([
    [0,0,1,1,0],
    [0,0,1,1,0],
    [0,0,9,0,0],
    [0,1,1,0,0],
    [0,1,1,0,0],
  ], dtype=np.uint8),
  np.array([
    [0,0,0,1,1],
    [0,0,1,1,0],
    [0,0,9,0,0],
    [0,1,1,0,0],
    [1,1,0,0,0],
  ], dtype=np.uint8),
  np.array([
    [0,0,0,0,1],
    [0,0,0,1,0],
    [0,0,9,0,0],
    [0,1,0,0,0],
    [1,0,0,0,0],
  ], dtype=np.uint8),
  np.array([
    [0,0,0,0,1],
    [0,0,0,1,1],
    [0,1,9,1,0],
    [1,1,0,0,0],
    [1,0,0,0,0],
  ], dtype=np.uint8),
  np.array([
    [0,0,0,0,0],
    [0,0,0,1,1],
    [1,1,9,1,1],
    [1,1,0,0,0],
    [0,0,0,0,0],
  ], dtype=np.uint8),
  np.array([
    [0,0,0,0,0],
    [0,0,0,0,0],
    [1,1,9,1,1],
    [0,0,0,0,0],
    [0,0,0,0,0],
  ], dtype=np.uint8),
  np.array([
    [0,0,0,0,0],
    [1,1,0,0,0],
    [1,1,9,1,1],
    [0,0,0,1,1],
    [0,0,0,0,0],
  ], dtype=np.uint8),
  np.array([
    [1,0,0,0,0],
    [1,1,0,0,0],
    [0,1,9,1,0],
    [0,0,0,1,1],
    [0,0,0,0,1],
  ], dtype=np.uint8),
  np.array([
    [1,0,0,0,0],
    [0,1,0,0,0],
    [0,0,9,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1],
  ], dtype=np.uint8),
  np.array([
    [1,1,0,0,0],
    [0,1,1,0,0],
    [0,0,9,0,0],
    [0,0,1,1,0],
    [0,0,0,1,1],
  ], dtype=np.uint8),
  np.array([
    [0,1,1,0,0],
    [0,1,1,0,0],
    [0,0,9,0,0],
    [0,0,1,1,0],
    [0,0,1,1,0],
  ], dtype=np.uint8),
)

def get_bitembed(gray, ang=4, apply_filter=True):
  hue, brightness = calc(gray)
  _, mask = cv2.threshold(brightness, 127, 65535, cv2.THRESH_BINARY)

  res = np.zeros(gray.shape, dtype=np.uint16)
  for x in range(12):
    lut = np.zeros((256, 1), dtype=np.uint8)
    for v in range(x*15-ang, x*15 +ang+1):
      lut[v%180][0] = 1
    bits = cv2.LUT(hue, lut)
    bits = cv2.bitwise_and(bits, mask)
    if apply_filter:
      bits = cv2.filter2D(bits, cv2.CV_16U, FILTERS[x])
      _, result = cv2.threshold(bits, 10, 1<<x, cv2.THRESH_BINARY)
    else:
      bits = bits.astype(np.uint16)
      _, result = cv2.threshold(bits, 0, 1<<x, cv2.THRESH_BINARY)
    res = cv2.bitwise_or(res, result)
  # print("res max,min,type", np.max(res), np.min(res), res.dtype)
  visualize.img(res, 'hsv', True)
  return res
