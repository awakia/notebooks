import matplotlib.pyplot as plt
import random
import cv2
import numpy as np

def random_color():
  return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)

def img(img, cmap=None, norm=None):
  plt.figure(figsize=(10,10), dpi=200)
  plt.imshow(img, cmap, norm)
  plt.show()

def with_found(img, found, plot=True):
  out = img.copy()
  for x, y, size in found:
    color = random_color()
    out = cv2.rectangle(out, (x, y), (x+size, y+size), color, 2)
  if plot:
    plt.figure(figsize=(10,10), dpi=200)
    plt.imshow(out)
    plt.show()
  return out

def with_lines(img, lines, plot=True):
  out = img.copy()
  for l in lines:
    color = random_color()
    out = cv2.line(out, tuple(l[0]), tuple(l[1]), color, 3)
  if plot:
    plt.figure(figsize=(10,10), dpi=200)
    plt.imshow(out)
    plt.show()
  return out

def with_line_group(img, line_group, plot=True):
  out = img.copy()
  for lines in line_group:
    color = random_color()
    for l in lines:
      out = cv2.line(out, tuple(l[0]), tuple(l[1]), color, 3)
  if plot:
    plt.figure(figsize=(10,10), dpi=200)
    plt.imshow(out)
    plt.show()
  return out

def with_polylines(img, polylines, color=None, thickness=1, circle=0, plot=True):
  polylines = np.array(polylines, dtype=np.int32)
  if color is None:
    color = random_color()
  out = cv2.polylines(img, [polylines], True, color, thickness)
  if circle > 0:
    for p in polylines:
      cv2.circle(out, tuple(p), circle, color, thickness=-1)
  if plot:
    plt.figure(figsize=(10,10), dpi=200)
    plt.imshow(out)
    plt.show()
  return out
