import cv2
import numpy as np

PATH = 'image/light2.png'
img = cv2.imread(PATH, cv2.IMREAD_UNCHANGED)
#img = cv2.resize(img, [400, 400])

img_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

img_remove = img_HLS[:, :, 1]
res = np.zeros([img.shape[0], img.shape[1], 4])
res[:, :, :3] = img
res[:, :, 3] = img_remove
cv2.imwrite("image/light2_alpha.png", res)
cv2.imshow("alpha", img_remove)
cv2.waitKey(0)
cv2.destroyAllWindows()
