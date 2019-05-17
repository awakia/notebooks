import glob
import math
import cv2
import numpy as np

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
  return bgr

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
  et = edge_type(gray)
  visualize.img(et)
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
