
import cv2
import pyflow
import numpy as np


def color_code(flow, maxmag=10):
    x, y = flow[:, :, 0].astype(np.float32), flow[:, :, 1].astype(np.float32)
    magnitude, angle = cv2.cartToPolar(x, y, angleInDegrees=True)
    magnitude = np.clip(magnitude, 0, maxmag) / maxmag

    hsv = np.ones((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    hsv[:, :, 0] = angle
    hsv[:, :, 1] = magnitude
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


im0 = cv2.imread('frame_0.png')
im1 = cv2.imread('frame_1.png')

im0_g = cv2.cvtColor(im0, cv2.COLOR_RGB2GRAY)
im1_g = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)

#f = pyflow.brox(im0_g / 255., im1_g / 255.)
f = pyflow.tvl1(im0_g / 255., im1_g / 255.)
cv2.imshow('brox', color_code(f))

cv2.waitKey(0)
