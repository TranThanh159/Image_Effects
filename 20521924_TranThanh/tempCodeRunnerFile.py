#Xử lí ảnh làm đầu vào cho model
    # frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    # #detect kết quả
    # result = model_landmarker.detect(frame_mp)
    # print(result)
    
    # #Lấy mask từ result ->ma trận 3 chiều
    # segmentation_mask = result.segmentation_masks[0].numpy_view()
    # mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
    # #Lấy các tọa độ các điểm từ kết quả
    # points = result.pose_landmarks
    
    # #Thực hiện blend
    # final = blend_with_mask(frame, eff_resize, mask, [points[0][0].x, points[0][0].y], 0.5)
    
    # #viết vào output
    # out.write(final)
    
    # #show kết quả
    # cv.imshow(final)