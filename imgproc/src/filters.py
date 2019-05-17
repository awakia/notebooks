import cv2

def resize(img, max_side=2000):
  h, w = img.shape[:2]
  side = max(h, w)
  shift = 0
  while (side >> shift) > max_side:
    shift += 1
  return cv2.resize(img, (w >> shift, h >> shift))

def rotate(img, angle, scale = 1.0):
  (h, w) = img.shape[:2]
  center = (w / 2, h / 2)

  # Perform the rotation
  M = cv2.getRotationMatrix2D(center, angle, scale)
  rotated = cv2.warpAffine(img, M, (w, h))

  return rotated
