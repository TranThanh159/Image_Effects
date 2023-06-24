import numpy as np
import cv2 as cv

#width, height
SIZE = (640, 480)
FPS = 60
NAME = 'video/input0.avi'

fourcc = cv.VideoWriter_fourcc('D', 'I', 'V', 'X')
out = cv.VideoWriter(NAME, fourcc, FPS, SIZE, isColor=True)

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret: 
        break
    
    frame = cv.resize(frame, SIZE)
    frame = np.flip(frame, axis=1)
    
    cv.imshow("live", frame)
    
    out.write(frame)
    
    if cv.waitKey(1) == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()
    
    
