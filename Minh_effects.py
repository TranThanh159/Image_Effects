import cv2
from PIL import Image 
import numpy as np
import math
class ImgProcess:
    def __init__(self, img) -> None:
        self.src = cv2.imread(img) 
        self.gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY) 
        self.h, self.w = self.src.shape[:2]  
    def oil(self):       #hiệu ứng ảnh tranh sơn dầu
        oilImg = np.zeros((self.h, self.w, 3), np.uint8)
        for i in range(2, self.h - 2):
            for j in range(2, self.w - 2):
                
                quant = np.zeros(8, np.uint8)
                
                for k in range(-2, 2):
                    for t in range(-2, 2):
                        level = int(self.gray[i + k, j + t] / 32)
    
                        quant[level] = quant[level] + 1
                valIndex = np.argmax(quant)

                for k in range(-2, 2):
                    for t in range(-2, 2):

                        if (valIndex * 32) <= self.gray[i + k, j + t] <= ((valIndex + 1) * 32):
                            (b, g, r) = self.src[i + k, j + t]
                            oilImg[i, j] = (b, g, r)
        return oilImg

    def old(self):        # hiệu ứng ảnh màu cũ
        oldImg = np.zeros((self.h, self.w, 3), np.uint8)
        for i in range(self.h):
            for j in range(self.w):

                b = 0.272 * self.src[i, j][2] + 0.534 * self.src[i, j][1] + 0.131 * self.src[i, j][0]
                g = 0.349 * self.src[i, j][2] + 0.686 * self.src[i, j][1] + 0.168 * self.src[i, j][0]
                r = 0.393 * self.src[i, j][2] + 0.769 * self.src[i, j][1] + 0.189 * self.src[i, j][0]
                if b > 255:
                    b = 255
                if g > 255:
                    g = 255
                if r > 255:
                    r = 255
                oldImg[i, j] = np.uint8((b, g, r))
        return oldImg
    
 
    # def pencil(self):          #hiệu ứng vẽ bút chì
    #     gray = self.gray
    #     neg = 255 - gray
    #     blur = cv2.GaussianBlur(neg, ksize=(21, 21), sigmaX=0, sigmaY=0)
    #     blend = cv2.divide(gray, 255 - blur, scale=256)
        
    #     return blend

    def pencil(self):
        gray = self.gray   
        neg = 255 - gray    
        kernel = np.ones((21,21),np.float32)/(21*21)
        blur = cv2.filter2D(neg,-1,kernel)
        blend = np.zeros((self.h, self.w), np.uint8)
        for i in range(self.h):
            for j in range(self.w):
                if (255 - blur[i, j]) == 0:
                    blend[i, j] = gray[i, j]
                else:
                    blend[i, j] = np.uint8(gray[i, j] / (255 - blur[i, j]) * 255)
    
        return blend


if __name__ == '__main__':
    process_types = ['oil','old','pencil']   
    for process_type in process_types:
        process = ImgProcess('base.jpg')  #input ảnh vào rồi chạy
        processed_img = getattr(process, process_type)()  
        cv2.imshow(process_type, processed_img) 
        cv2.waitKey(delay=0) 
