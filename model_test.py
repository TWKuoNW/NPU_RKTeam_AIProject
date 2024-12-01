from ultralytics import YOLO
import cv2  
import numpy as np

cap = cv2.VideoCapture('http://100.81.241.109:8001/video') #'D:/video/TestVideo.avi''http://100.81.241.109:8002/video'
model = YOLO('D:\\智慧養殖專題\\Code\\AI\\models\\ShrimpModelV1.pt')  

def nothing(x):
    pass

cv2.namedWindow('ImageMaster')
cv2.resizeWindow('ImageMaster', 600, 160)
cv2.createTrackbar('size', 'ImageMaster', 25, 100, nothing)
cv2.createTrackbar('blur', 'ImageMaster', 1, 21, nothing)
cv2.createTrackbar('speed', 'ImageMaster', 0, 3000, nothing)

ret, origin_frame = cap.read()  # 讀取一幀

while(cap.isOpened()):
    ret, frame = cap.read()  # 讀取一幀
    if(not ret):
        break  # 如果無法讀取，跳出循環

    img_size = cv2.getTrackbarPos('size', 'ImageMaster')
    blur_value = cv2.getTrackbarPos('blur', 'ImageMaster')
    speed_value = cv2.getTrackbarPos('speed', 'ImageMaster')
    if(blur_value % 2 == 0):
        blur_value += 1  # 如果是偶數，則加1變成奇數
    
    y, x, _ = origin_frame.shape
    x = int(x / 100 * img_size)
    y = int(y / 100 * img_size)
    frame = cv2.resize(frame, (x, y))  # 調整畫面的大小
    # 定義銳化濾波器
    sharpening_filter = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])

    # 應用銳化濾波器
    frame2 = cv2.filter2D(frame, -1, sharpening_filter)
    
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 將幀轉換成 RGB
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_value, blur_value), 2)
    img = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)
    

    results = model([img])

    for result in results:
        sourceTensor = result.boxes.data
        data = sourceTensor.clone().detach().tolist()        
        formatted_data = [[int(value) if idx == None else round(value, 3) for idx, value in enumerate(inner_list)] for inner_list in data]
        # print(formatted_data)
        confidence_list = []
        for i in range(len(formatted_data)):
            x_min, y_min, x_max, y_max, _, cls_id = map(int, formatted_data[i]) # 取得物件的座標
            _, _, _, _, conf, _ = map(float, formatted_data[i]) # 取得物件的信心度
            confidence_list.append(conf)

            if(cls_id == 0):
                cv2.putText(frame, f'shrimp:{(conf*100):.2f}%', (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # 在物件上顯示可信度
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1)  # 畫矩形框 
            
        max_confidence = max(confidence_list)
        max_confidence_index = confidence_list.index(max_confidence)
        x_min, y_min, x_max, y_max, _, _ = map(int, formatted_data[max_confidence_index]) # 取得物件的座標
        cropped_region = frame2[y_min:y_max,x_min:x_max]
        y, x, _ = cropped_region.shape
        print(cropped_region.shape)
        cropped_region = cv2.resize(cropped_region, (x*2, y*2))
        


    cv2.imshow('gray', gray)   
    cv2.imshow('blur', blur)       
    cv2.imshow('img', img)
    cv2.imshow('Video', frame)  # 顯示處理後的影像幀
    cv2.imshow('Video2', frame2)
    cv2.imshow('Cropped Region', cropped_region)

    if cv2.waitKey(speed_value+1) == ord('q'):  # 如果按下 'q' 鍵，則退出
        break
    
cap.release()  # 釋放 cap
cv2.destroyAllWindows()  # 關閉所有 OpenCV 窗口
print("Done!")