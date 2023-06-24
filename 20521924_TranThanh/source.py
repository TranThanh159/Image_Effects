import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

#BASE_SIZE = (width, height), VGA, 30fps
BASE_SIZE = (640, 480)
EFF_SIZE = (200, 200)
FPS = 60
#EFF là neon circle light
EFF_PATH = 'image/neon_yellow_circle.png' 
BACKGOUND_PATH = 'image/background.png'
MODEL_PATH = 'model/pose_landmarker_lite.task'

HAND_CRYSTAL_PATH = 'image/crystal2_alpha.png'
HAND_CIRCLE_PATH = 'image/neon_blue_circle.png'
HAND_LIGHT_PATH = 'image/light2_alpha.png'

INPUT_VIDEO_PATH = 'video/input.avi'
OUT_VIDEO_PATH = 'video/output.avi'                                                                                                                                                                                       

THRESH_BASE_MASK = 125

#hàm blend ảnh có mask theo alpha và effect là ảnh BGRA
def blend_with_mask(base: np.ndarray, effect: np.ndarray, mask: np.ndarray, normalized_central_point=[0, 0], thresh=THRESH_BASE_MASK):
  h_base, w_base = base.shape[0: 2]
  h_eff, w_eff = effect.shape[0: 2]

  # x, y start apply effect
  x_start = round(normalized_central_point[0]*w_base) - w_eff//2
  y_start = round(normalized_central_point[1]*h_base) - h_eff//2

  x_end = x_start + w_eff
  y_end = y_start + h_eff

  x_start = max(x_start, 0)
  y_start = max(y_start, 0)
  x_end = min(x_end, w_base)
  y_end = min(y_end, h_base)

  #blend
  res = base.copy()
  
  blend_base = base[y_start: y_end, x_start: x_end]
  blend_mask = mask[y_start: y_end, x_start: x_end]
  #ảnh eff có xu hướng bị mất bên dưới khi vượt quá
  blend_eff_bgr = effect[0: (y_end - y_start), 0: (x_end - x_start), :3]
 
  
  #blend alpha 0-1
  x = effect[0: (y_end - y_start), 0: (x_end - x_start), 3]
  blend_eff_alpha = np.repeat(x, 3, axis=1)
  blend_eff_alpha = np.reshape(blend_eff_alpha, blend_eff_bgr.shape)
  blend_eff_alpha = np.asarray(blend_eff_alpha, dtype=np.float32)
  blend_eff_alpha = blend_eff_alpha/255.0
  
  blend_area = blend_base.copy()
  
  #blend thresh 0-255 giúp xử lí nhiễu ảnh khi blend
  condition = (blend_mask <= thresh)
  blend_area[condition] = blend_base[condition]*(1-blend_eff_alpha[condition]) + blend_eff_bgr[condition]*blend_eff_alpha[condition]

  res[y_start: y_end, x_start: x_end] = blend_area

  return res

#định nghĩa hàm vẽ landmark
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

#Tạo các thông số cho model
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options = BaseOptions(model_asset_path=MODEL_PATH),
    running_mode = VisionRunningMode.IMAGE,
    num_poses = 1,
    min_pose_detection_confidence = 0.5,
    min_pose_presence_confidence = 0.5,
    min_tracking_confidence = 0.5,
    output_segmentation_masks = True
)
              
#Tải model về sử dụng
model_landmarker = vision.PoseLandmarker.create_from_options(options)





#Đọc ảnh eff RGBA
eff = cv2.imread(EFF_PATH, cv2.IMREAD_UNCHANGED)
background = cv2.imread(BACKGOUND_PATH)
bg_resize = cv2.resize(background, BASE_SIZE)

hand_crystal = cv2.imread(HAND_CRYSTAL_PATH, cv2.IMREAD_UNCHANGED)
hand_circle = cv2.imread(HAND_CIRCLE_PATH, cv2.IMREAD_UNCHANGED)
hand_light = cv2.imread(HAND_LIGHT_PATH, cv2.IMREAD_UNCHANGED)
#cloud = cv2.imread(CLOUD_PATH, cv2.IMREAD_UNCHANGED)

eff = cv2.resize(eff, EFF_SIZE)
hand_circle_2 = cv2.resize(hand_circle, [np.max(BASE_SIZE), np.max(BASE_SIZE)])

hand_crystal = cv2.resize(hand_crystal, EFF_SIZE)
hand_circle = cv2.resize(hand_circle, EFF_SIZE)
hand_light = cv2.resize(hand_light, EFF_SIZE)

#cloud = cv2.resize(cloud, EFF_SIZE)
#Vòng sáng to-nhỏ theo tỉ lệ từ 25% -> 100%
number_of_light = 8
hand_circle_range = [x/(number_of_light*4) for x in range(number_of_light, number_of_light*4 + 1)]
hand_circle_index = 0

#vòng tròn ánh sáng lớn theo tỉ lệ từ 10% - 100%
number_of_light_2 = 5
hand_circle_range_2 = [x/(number_of_light_2*10) for x in range(number_of_light_2, number_of_light_2*10 + 1)]
hand_circle_index_2 = 0

#Đọc video input và  xủ lí
video = cv2.VideoCapture(INPUT_VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
out = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, FPS, BASE_SIZE, isColor=True)

#test dilate, erode 
# out_mask = cv2.VideoWriter('video/output_mask.avi', fourcc, FPS, BASE_SIZE, isColor=True)
# out_dilate = cv2.VideoWriter('video/output_dilate.avi', fourcc, FPS, BASE_SIZE, isColor=True)
# out_erode = cv2.VideoWriter('video/output_erode.avi', fourcc, FPS, BASE_SIZE, isColor=True)
kernel = np.ones([4, 4])

#Đếm frame
number = 0
smooth_step = 3

while video.isOpened(): 
    frames = []
    ret = True
    for i in range(smooth_step):
      ret, frame = video.read()
      
      if not ret: break
      frames.append(frame)
    
    if not ret: break
    
    
    #resize khung hình
    frames_resize = [cv2.resize(frame, BASE_SIZE) for frame in frames]
    
    # #phát video
    # cv2.imshow("Video", frame)
      
      #Thay kiểu dữ liệu ảnh làm đầu vào cho model
    frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frames_resize[0])
    
    #detect kết quả
    result = model_landmarker.detect(frame_mp)
    
    #Lấy mask từ result ->ma trận 3 chiều mask
    mask = []
    if result.segmentation_masks != None:
      segmentation_mask = result.segmentation_masks[0].numpy_view()
      mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
      mask = np.uint8(mask)
  
      # out_mask.write(mask)
    
      # mask_dilate = cv2.dilate(mask, kernel)
      # out_dilate.write(mask_dilate)
      
      # mask_erode = cv2.erode(mask, kernel)
      # out_erode.write(mask_erode)
    
    
    #Lấy các tọa độ các điểm từ kết quả
    points = result.pose_landmarks
    

    for frame in frames_resize:
      final = frame.copy()
      #trigger kích hoạt neon circle + thay background
      #cổ tay cao hơn vai (tay trái so với người trong ảnh)
      if len(points) != 0:
        # print(points)
        if points[0][15].y < points[0][11].y:
          #Thay background
          thresh_replace = THRESH_BASE_MASK
          frame_replace_bg = bg_resize.copy()
          frame_replace_bg[mask >= thresh_replace ] = frame[mask >= thresh_replace]
          
          #Lấy 2 điểm vai index 11, 12
          #neon circle sẽ phóng to thu nhở theo tỉ lệ cơ thể
          mean_eyes = [(points[0][2].x + points[0][5].x)/2, (points[0][2].y + points[0][5].y)/2]
          mean_shoulders = [(points[0][11].x + points[0][12].x)/2, (points[0][11].y + points[0][12].y)/2]
          
          scale = 3.5
          distance_eyes_shoulders = round(((mean_eyes[0]-mean_shoulders[0])**2 + (mean_eyes[1]-mean_shoulders[1])**2)**(1/2)*scale*BASE_SIZE[1])
          
          eff_resize = cv2.resize(eff, [distance_eyes_shoulders, distance_eyes_shoulders])
          
          # #effect dưới chân
          # left_knee = points[0][25]
          # right_knee = points[0][26]
          # distance_knee = round(((left_knee[0]-right_knee[0])**2 + (left_knee[1]-right_knee[1])**2)**(1/2)*scale)
          # mean_knees = [(left_knee[0] + right_knee[0])/2, (left_knee[1] + right_knee[1])/2]
          # #cloud_resize = cv2.resize(cloud, [distance_knee, distance_knee])
          
          #Effect trên tay
          hand_circle_index = (hand_circle_index + 1) % len(hand_circle_range)
          dis_crystal = round(distance_eyes_shoulders*0.40)
          dis_circle = round(distance_eyes_shoulders*0.80*hand_circle_range[hand_circle_index])
          dis_light = round(distance_eyes_shoulders*0.80)
          hand_center = [(points[0][15].x + points[0][17].x + points[0][19].x)/3, (points[0][15].y + points[0][17].y + points[0][19].y)/3]
          hand_circle_resize = cv2.resize(hand_circle, [dis_circle, dis_circle])
          hand_crystal_resize = cv2.resize(hand_crystal, [dis_crystal, dis_crystal])
          hand_light_resize = cv2.resize(hand_light, [dis_light, dis_light])
          
          #Thực hiện blend trên tay
          #Nếu cổ tay cao hơn đầu (tính bằng mean_eyes) thì xuất hiện thêm ánh sáng object
          
          
          final0 = blend_with_mask(frame_replace_bg, hand_light_resize, np.zeros(frame_replace_bg.shape), hand_center)
          
          final1 = blend_with_mask(final0, hand_crystal_resize, np.zeros(final0.shape), hand_center)
          #final2 = blend_with_mask(final1, cloud_resize, np.zeros(final1.shape), mean_knees)
          
          #Thực hiện blend vòng sáng
          final = blend_with_mask(final1, eff_resize, mask, mean_eyes)
          
          #Thêm vòng 2 sáng cho crystal khi cổ tay cao hơn đầu
          if points[0][15].y < mean_eyes[1]:
            final = blend_with_mask(final, hand_circle_resize, np.zeros(final.shape), hand_center)
                
            hand_circle_index_2 = (hand_circle_index_2 + 1) % len(hand_circle_range_2)
            dis_circle_2 = round(np.max(BASE_SIZE)*hand_circle_range_2[hand_circle_index_2])
            hand_circle_2_resize  = cv2.resize(hand_circle_2, [dis_circle_2, dis_circle_2])
            final = blend_with_mask(final, hand_circle_2_resize, np.zeros(final.shape), hand_center)
            
            
      #viết vào output
      out.write(final)
      
      # #show kết quả
      cv2.imshow("final", final)
    print("Processing with the frame number: ", number)
    number += smooth_step
    
    if cv2.waitKey(1) == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()