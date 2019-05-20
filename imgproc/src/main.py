import glob
import cv2

import edge_type
import mark_detector
import filters
import visualize

# cmap reference: https://matplotlib.org/examples/color/colormaps_reference.html
def proc(img):
  print(img.shape)
  visualize.img(img, "gray")
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # visualize.img(gray, "gray")
  edge_type.get_bitembed(gray, apply_filter=False) # to visualize original
  bits = edge_type.get_bitembed(gray)
  ditector = mark_detector.TriangleDetector(bits)
  # ditector = mark_detector.CircleDetector(bits)
  found = ditector.detect(6)
  visualize.with_found(img, found)
  visualize.img(img)

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
