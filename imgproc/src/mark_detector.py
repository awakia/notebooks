import cv2
from abc import abstractmethod

USED_FLAG = 0x8000

class MarkDetector:
  def __init__(self, bits):
    self.layers = [bits]
    self.found = [] # (x, y, size)

  def add_layer(self):
    n = len(self.layers)
    N = 1 << (n-1)
    prev = self.layers[-1]
    print("adding layer", n+1, prev.shape)
    if N >= prev.shape[0] or N >= prev.shape[1]:
      return False
    layer = cv2.bitwise_or(
      cv2.bitwise_or(prev[:-N,:-N], prev[N:,:-N]),
      cv2.bitwise_or(prev[:-N,N:], prev[N:,N:])
    )
    self.layers.append(layer)
    return True

  def check_layer(self):
    prev = self.layers[-1]
    h, w = prev.shape
    for y in range(h):
      for x in range(w):
        val = prev[y, x]
        # if USED_FLAG & val == USED_FLAG: continue
        if self.match(val):
          # val |= USED_FLAG
          self.found.append((x, y, 1 << (len(self.layers)-1)))

  def detect(self, max_layer = None):
    while True:
      if max_layer is not None and len(self.layers) >= max_layer: break
      if not self.add_layer(): break
      self.check_layer()
    return self.found

  @abstractmethod
  def match(self, val):
    raise NotImplementedError()

class TriangleDetector(MarkDetector):
  # def __init__(self, bits):
  #   super().__init__(bits)

  def match(self, val):
    # return (
    #   val & 0b100010001000 == 0b100010001000 or
    #   val & 0b001000100010 == 0b001000100010 or
    #   val & 0b010001000100 == 0b010001000100 or
    #   val & 0b100010001000 == 0b100010001000
    # )
    return (
      val == 0b100010001000 or
      val == 0b001000100010 or
      val == 0b010001000100 or
      val == 0b100010001000
    )
