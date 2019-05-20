import glob
import math
import cv2
import numpy as np
import mark_detector

import filters
import visualize

def edge_type(gray):
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
  saturation = np.full(gray.shape, 255, dtype=np.uint8)
  hsv = np.dstack((hue, saturation, brightness))
  bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
  visualize.img(bgr)
  return hue, brightness

def bitembed_edge_type(hue, brightness, ang=3, thresh=128):
  lut = np.zeros((256, 1), dtype=np.uint16) # or 65536
  for x in range(12):
    for v in range(x*15-ang, x*15 +ang+1):
      lut[v%180][0] |= 1 << x # 1 + x
  br = brightness.astype(np.uint16)
  th, mask = cv2.threshold(br, 127, 65535, cv2.THRESH_BINARY)
  # visualize.img(mask)
  bits = cv2.LUT(hue, lut)
  res = cv2.bitwise_and(bits, mask)
  # print("res max,min,type", np.max(res), np.min(res), res.dtype)
  visualize.img(res, 'hsv', True)
  return res

def get_lines(gray):
  lsd = cv2.createLineSegmentDetector()
  lines, width, prec, nfa = lsd.detect(gray)
  return np.reshape(lines, (len(lines), 2, 2))


# cmap reference: https://matplotlib.org/examples/color/colormaps_reference.html
# https://en.wikipedia.org/wiki/Sobel_operator
def proc(img):
  print(img.shape)
  visualize.img(img, "gray")
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # visualize.img(gray, "gray")
  hue, brightness = edge_type(gray)
  bits = bitembed_edge_type(hue, brightness)
  # ditector = mark_detector.TriangleDetector(bits)
  ditector = mark_detector.CircleDetector(bits)
  found = ditector.detect(5)
  visualize.with_found(img, found)
  visualize.img(img)

  # lines = get_lines(gray)
  # visualize.with_lines(img, lines)

def main():
  # path = "data/all/*.png"
  path = "data/sample/*.png"
  for i, imgfile in enumerate(sorted(glob.glob(path))):
    print(imgfile)
    # img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    proc(img)

if __name__ == '__main__':
  main()
