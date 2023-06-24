import numpy as np
import cv2

cap = cv2.VideoCapture('image/falling_petal.gif')

number = 0

while True:
    ret, frame = cap.read(1)
    
    if not ret: break
    number += 1
    frame = np.flip(frame, axis=0)
    cv2.imshow("gif", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
print(number)


ret, frame = cap.read()
cv2.imshow("img", frame)