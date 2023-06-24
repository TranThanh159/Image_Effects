import numpy as np
import cv2

mask = cv2.VideoCapture('video/output_mask.avi')
dilate = cv2.VideoCapture('video/output_dilate.avi')
erode = cv2.VideoCapture('video\output_erode.avi')

ret, frame_m = mask.read()
ret, frame_d = dilate.read()
ret, frame_e = erode.read()

cv2.imwrite('output_mask.png', frame_m)
cv2.imwrite('output_dilate.png', frame_d)
cv2.imwrite('output_erode.png', frame_e)

