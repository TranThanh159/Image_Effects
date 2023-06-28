import cv2
import mediapipe as mp
import numpy as np
import math

# Khởi tạo mediapipe Hands và các biến cần thiết
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

# Khởi tạo video capture từ camera và video file
video = cv2.VideoCapture(0)
vid = cv2.VideoCapture('/Users/phongminh/20522217_PhongLaiBaoMinh/source/video.mp4')

# Thiết lập kích thước video
video.set(3, 500)
video.set(4, 500)


# Load các hình ảnh của magic circle
img_1 = cv2.imread('/Users/phongminh/20522217_PhongLaiBaoMinh/source/magic_circle_1.png', -1)
img_2 = cv2.imread('/Users/phongminh/20522217_PhongLaiBaoMinh/source/magic_circle_ccw.png', -1)
img_3 = cv2.imread('/Users/phongminh/20522217_PhongLaiBaoMinh/source/magic_circle_2.png', -1)
img_4 = cv2.imread('/Users/phongminh/20522217_PhongLaiBaoMinh/source/magic_circle_cw.png', -1)

deg=0 # góc xoay ảnh ban đầu


# Hàm vẽ đường thẳng
def draw_line(p1, p2, size=5):
    cv2.line(img, p1, p2, (50,50,255), size)
    cv2.line(img, p1, p2, (255, 255, 255), round(size / 2))

# tọa độ điểm trên ngón tay
def position_data(lmlist):
    global wrist, thumb_top, index_top, index_root, midle_top, ring_top, pinky_top
    wrist = (lmlist[0][0], lmlist[0][1])       
    thumb_top = (lmlist[4][0], lmlist[4][1])    
    index_root = (lmlist[5][0], lmlist[5][1])
    index_top = (lmlist[8][0], lmlist[8][1])    
    midle_top = (lmlist[12][0], lmlist[12][1])  
    ring_top  = (lmlist[16][0], lmlist[16][1])  
    pinky_top = (lmlist[20][0], lmlist[20][1])  
    
# tính khoảng cách
def length2point(p1,p2):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    lenght = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1.0 / 2)
    return lenght

# Tìm cos đầu 3 điểm 
def cos3(a, b, c):
    xa, ya, xb, yb, xc, yc = a[0], a[1], b[0], b[1], c[0], c[1]
    x1 = xb - xa
    y1 = yb - ya
    x2 = xc - xa
    y2 = yc - ya
    cosin = (x1*x2 + y1*y2) / (math.sqrt(x1*x1 + y1*y1)*math.sqrt(x2*x2 + y2*y2))
    return cosin
    
#Thay đổi fg vs bg
def transparent(targetImg, x, y, size=None):
    if size is not None:
        targetImg = cv2.resize(targetImg, size)
    # Tạo ảnh buffer có tỉ lệ tương tự như ảnh gốc
    newFrame = img.copy()
    # Tách lớp từ ảnh target
    b, g, r, a = cv2.split(targetImg)
    # Tạo mask từ lớp alpha
    mask = a
    # Bỏ hệ số alpha
    target_color = cv2.merge((b, g, r))
    h, w, _ = target_color.shape
    # Chọn vùng của ảnh gốc để chèn ảnh
    region = newFrame[y:y + h, x:x + w] 
    # Ảnh FG và BG
    img1_bg = cv2.bitwise_and(region.copy(), region.copy(), mask = cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(target_color, target_color, mask = mask)
    # Thay BG bằng FG
    newFrame[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)

    return newFrame

count = 0
countEnergy = 0
FullEnergy = check_effect = False
effect = 0
while True:
    ret, frame = video.read() # Đọc frame từ video capture
    img = cv2.flip(frame, 1)
    rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgbimg)
    if result.multi_hand_landmarks:  # Vẽ các điểm và đường nối giữa chúng trên ảnh
        count = 0
        xr = yr = dr = cosr = xl = yl = dl = cosl = 0 
        
        #Hand1
        thumbr, indexr = (0, 0)

        for hand in result.multi_hand_landmarks:
            count +=1
            lmList=[]

            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                x_, y_=int(lm.x*w), int(lm.y*h)
                lmList.append([x_, y_])
            position_data(lmList)
            
            draw_line(wrist, thumb_top)
            draw_line(wrist, index_top)
            draw_line(wrist, midle_top)
            draw_line(wrist, ring_top)
            draw_line(wrist, pinky_top)
            
            draw_line(thumb_top, index_top)
            draw_line(thumb_top, midle_top)
            draw_line(thumb_top, ring_top)
            draw_line(thumb_top, pinky_top)

            wrist_index = length2point(wrist, index_top)
            wrist_index2 = length2point(wrist, index_root)
            wrist_midle = length2point(wrist, midle_top)
            wrist_ring = length2point(wrist, ring_top)
            wrist_pinky = length2point(wrist, pinky_top)
            wrist_3tail = (wrist_midle + wrist_ring + wrist_pinky) / 3
            ratio = wrist_index2/wrist_3tail
            
            if ((ratio > 1) & (wrist_index2 < wrist_index)):
                shield_size = 1.25
                diameter = round(length2point(thumb_top, index_top) * shield_size)
                x = round((thumb_top[0] + index_top[0])/2 - (diameter / 2))
                y = round((thumb_top[1] + index_top[1])/2 - (diameter / 2))
                h, w, c = img.shape

                #xử lý hình bàn tay ở góc hình
                if x < 0:   x = 0
                elif x > w: x = w
                if y < 0:   y = 0
                elif y > h: y = h

                if x + diameter > w:    diameter = w - x
                if y + diameter > h:    diameter = h - y

                #Đếm time 
                if (xr == x) & (yr == y):
                    xr == 0
                    yr == 0

                if count == 1:
                    xr = x
                    yr = y
                    dr = diameter
                    thumbr = thumb_top
                    indexr = index_top
                else: 
                    xl = x
                    yl = y
                    dl = diameter
                    if (dr != 0) & (dl != 0):
                        cosl = cos3(thumb_top, index_top, (xr + dr, yr + dr))
                        cosr = cos3(thumbr, indexr, (xl + dl, yl + dl))
            else: 
                countEnergy = 0

            xx = round((xr + xl)/2)
            yy = round((yr + yl)/2)
            dd = round((dr + dl)/2)
            addpix = 0
            if count == 2:
                coss = (abs(cosl) + abs(cosr))/2 - 0.1
                if coss > 0:
                    addpix = round(110 * coss)

            # Thiết kế xoay
            shield_size = dd, dd
            ang_vel = 5.0 #Tốc độ xoay
            deg = deg + ang_vel
            if deg > 360: deg = 0
            height, width, col = img_1.shape # vòng tròn
            cen = (width // 2, height // 2) # điểm ở trung tâm
            M1 = cv2.getRotationMatrix2D(cen, round(deg), 1.0) # xoay thuận
            M2 = cv2.getRotationMatrix2D(cen, round(360 - deg), 1.0) #xoay nghịch

            if (dd != 0) & (xr != 0 | yr != 0) & (xl != 0 | yl != 0) & (count == 2):
                # Tạo vòng tròn bình thường
                if (countEnergy <30):
                    countEnergy += 1
                    rotated1 = cv2.warpAffine(img_1, M1, (width, height))
                    rotated2 = cv2.warpAffine(img_2, M2, (width, height))
                    img = transparent(rotated1, xx, yy, shield_size)
                    img = transparent(rotated2, xx, yy, shield_size)  
                
                #chuyển hiệu ứng vòng tròn
                if (countEnergy >= 30):
                    FullEnergy = False
                    if (addpix < 90):
                        rotated1 = cv2.warpAffine(img_1, M1, (width, height))
                        rotated2 = cv2.warpAffine(img_2, M2, (width, height))

                        img = transparent(rotated1, xx, yy, shield_size)
                        img = transparent(rotated2, xx, yy, shield_size)  
                    if (addpix > 89):
                        rotated3 = cv2.warpAffine(img_3, M1, (width, height))
                        rotated4 = cv2.warpAffine(img_4, M2, (width, height))

                        img = transparent(rotated3, xx, yy, shield_size)
                        img = transparent(rotated4, xx, yy, shield_size)  
                        FullEnergy = True
    #Hiệu ứng video
    if FullEnergy:
        check_effect = False
        if wrist_index2 * 1.25 < wrist_pinky:
            check_effect = True
        
        if check_effect:
            height, width, channels = img.shape
            ret_val, flash_frame = vid.read()
            
            if not flash_frame is None:
                flash_frame = cv2.resize(flash_frame, (width, height))
                img = cv2.add(img, flash_frame)
    cv2.imshow("Image",img)

    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()