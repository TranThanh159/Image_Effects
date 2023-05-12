# import numpy as np
# import cv2
# import random

#dissolve modes
DISSOLVE_RANDOM_3TIMES = 1 
DISSOLVE_RANDOM_1TIMES = 2

#soft light modes
SOFT_LIGHT_PHOTOSHOP = 1
SOFT_LIGHT_PETOP = 2
SOFT_LIGHT_ILLUSION_HU = 3
SOFT_LIGHT_W3C = 4

#dodge modes
DODGE_SCREEN = 1
DODGE_COLOR_DODGE = 2
DODGE_LINEAR_DODGE = 3
DODGE_DIVIDE = 4

#burn modes
BURN_MULTIPLY = 1
BURN_COLOR_BURN = 2
BURN_LINEAR_BURN = 3

def show(img, text='image', wait=0):
    cv2.imshow(text, img)
    cv2.waitKey(wait)

#Chuyển từ int 0-255 sang float 0-1
def convert2float(layer):
    max = np.max(layer)
    
    if max > 1:
        res = layer/255
    return res

#Giới hạn giá trị trong khoảng chỉ chạy từ lie[0]-lie[1]
def limit(layer, lie=[0, 255]):
    min = np.full(layer.shape, lie[0])
    max = np.full(layer.shape, lie[1])
    
    res = np.fmin(layer, max)
    res = np.fmax(res, min)
    return res
    
#Xử lí các trường hợp có thể xuất hiện phép chia cho 0
def fix_zero_case(layer):
    input = convert2float(layer)
    
    #chuyển 0 -> 1/255(cận số 0), 1 -> 254/255(cận số 1)
    res = limit(input, [1/255, 254/255])
    return res
    
#blend theo alpha, alpha mức độ 'xâm lấn' lên ảnh gốc(base)
def normal_alpha(base, blend, alpha):
    A = convert2float(base)
    B = convert2float(blend)
    
    res = (1 - alpha)*A + alpha*B
    return res

#blend theo hướng random ngẫu nhiên đồng đều
def dissolve(base, blend, type = DISSOLVE_RANDOM_3TIMES):
    A = convert2float(base)
    B = convert2float(blend)
    h = base.shape[0]
    w = base.shape[1]
    
    if type == DISSOLVE_RANDOM_1TIMES:
        y = lambda x: [x, x, x]
        opacity = np.asarray([y([random.random()]) for x in range(h*w)])
        opacity = np.reshape(opacity, [h, w, 3])
        
    elif type == DISSOLVE_RANDOM_3TIMES:
        opacity = np.asarray([random.random() for x in range(h*w*3)])
        opacity = np.reshape(opacity, [h, w, 3])
    else:
        raise TypeError("Wrong type")
        
    res = (1-opacity)*A + opacity*B
    return res
    
#multiply làm ảnh có xu hướng tối đi
def multiply(base, blend):
    A = convert2float(base)
    B = convert2float(blend)
    
    res = A*B
    return res

#screen làm ảnh có xu hướng sáng lên
def screen(base, blend):
    A = convert2float(base)
    B = convert2float(blend)
    
    res = 1 - (1-A)*(1-B)
    return res

#overlay kết hợp giữa multiply và screen, tối->càng tối, sáng->càng sáng
def overlay(base, blend):
    A = convert2float(base)
    B = convert2float(blend)
    
    res = A.copy()
    res[A < 0.5] = 2*A[A < 0.5]*B[A < 0.5]
    res[A >= 0.5] = 1 - 2*(1 - A[A >= 0.5])*(1 - B[A >= 0.5])
    return res

#hardlight ngược lại so với overlay, khi B < 0.5 và B >= 0.5
def hardlight(base, blend):
    A = convert2float(base)
    B = convert2float(blend)
    
    res = A.copy()
    res[B < 0.5] = 1 - 2*(1 - A[B < 0.5])*(1 - B[B < 0.5])
    res[B >= 0.5] = 2*A[B >= 0.5]*B[B >= 0.5]
    return res

#softlight blend theo hướng nhẹ nhàng từ tốn hơn
def softlight(base, blend, type=SOFT_LIGHT_PHOTOSHOP):
    A = convert2float(base)
    B = convert2float(blend)
    
    if type == SOFT_LIGHT_PHOTOSHOP:
        res = A.copy()
        res[B < 0.5] = 2*A[B < 0.5]*B[B < 0.5] + A[B < 0.5]**2*(1-2*B[B < 0.5])
        res[B >= 0.5] = 2*A[B >= 0.5]*(1 - B[B >= 0.5]) + (A[B >= 0.5]**(1/2))*(2*B[B >= 0.5] - 1)    
    
    elif type == SOFT_LIGHT_PETOP:
        res = (1 - 2*B)*A**2 + 2*B*A
    
    elif type == SOFT_LIGHT_ILLUSION_HU:
        res = A**(2**(2*(0.5 - B)))
    
    elif type == SOFT_LIGHT_W3C:
        g_w3c = A.copy()
        g_w3c[A <= 0.25] = ((16*A[A <= 0.25] - 12)*A[A <= 0.25] + 4)*A[A <= 0.25]
        g_w3c[A > 0.25] = A[A > 0.25]**(1/2)
        
        f_w3c = A.copy()
        f_w3c[B <= 0.5] = A[B <= 0.5] - (1 - 2*B[B <= 0.5])*A[B <= 0.5]*(1 - A[B<= 0.5])
        f_w3c[B > 0.5] = A[B > 0.5] + (2*B[B > 0.5] - 1)*(g_w3c[B > 0.5] - A[B > 0.5])
         
        res = f_w3c   
    else:
        raise TypeError("Wrong type")
        
    return res

#trắng hơn
def dodge(base, blend, type=DODGE_COLOR_DODGE):
    A = convert2float(base)
    B = convert2float(blend)
    
    if type == DODGE_SCREEN:
        res = screen(A, B)
    
    elif type == DODGE_COLOR_DODGE:
        #fix chia số 0
        B = limit(B, [0, 254/255])
        
        res = A/(1-B)
    
    elif type == DODGE_LINEAR_DODGE:
        res = (A + B)
        res = limit(res, [0, 1])
    
    elif type == DODGE_DIVIDE:
        res = divide(A, B)
    else:
        raise TypeError("Wrong type")
        
    return res

#đen hơn
def burn(base, blend, type=BURN_COLOR_BURN):
    A = convert2float(base)
    B = convert2float(blend)
    
    if type == BURN_MULTIPLY:
        res = multiply(A, B)
        
    elif type == BURN_COLOR_BURN:
        #fix chia số 0
        B = limit(B, [1/255, 1])
        
        res = 1 - (1 - A)/B
        
    elif type == BURN_LINEAR_BURN:
        res = A + B - 1
        
        #fix số âm
        res = limit(res, [0, 1])
    
    return res

#color dodge b>0.5, color burn otherwise
def vivid_light(base, blend):
    A = convert2float(base)
    B = convert2float(blend)
    
    #fix chia số 0
    B = limit(B, [1/255, 254/255])
    
    res = A.copy()
    res[B > 0.5] = A[B > 0.5]/(1 - B[B > 0.5])
    res[B <= 0.5] = 1 - (1 - A[B <= 0.5])/B[B <= 0.5]
    
    return res

#linear dodge b>0.5, linear burn otherwise
def linear_light(base, blend):
    A = convert2float(base)
    B = convert2float(blend)

    res = A.copy()
    res[B < 0.5] = A[B < 0.5] + B[B < 0.5]
    res[B >= 0.5] = A[B >= 0.5] + B[B >= 0.5] - 1
    
    #fix số âm và vượt ngưỡng 1
    res = limit(res, [0, 1])
    return res
  
#divide  
def divide(base, blend):
    A = convert2float(base)
    B = convert2float(blend)
    
    #fix chia số 0
    B = limit(B, [1/255, 1])
    
    res = A/B
    return res

#addition
def addition(base, blend):
    A = convert2float(base)
    B = convert2float(blend)
    res = A + B
    
    #fix vượt ngưỡng 1
    res = limit(res, [0, 1])
    return res

#subtract
def subtract(base, blend):
    A = convert2float(base)
    B = convert2float(blend)
    res = A - B
    
    #fix số âm
    res = limit(res, [0, 1])
    return res

#abs(A, B)
def difference(base, blend):
    A = convert2float(base)
    B = convert2float(blend)
    res = np.abs(A - B)
    return res

#fmin(A, B)
def darken(base, blend):
    A = convert2float(base)
    B = convert2float(blend)
    res = np.fmin(A, B)
    return res

#fmax(A, B)
def lighten(base, blend):
    A = convert2float(base)
    B = convert2float(blend)
    res = np.fmax(A, B)
    return res


