import cv2
from roboflow import Roboflow

# 初始化 Roboflow 模型
rf = Roboflow(api_key="YxEHLn2gA26I3JvNFlua")
project = rf.workspace().project("goldfish-trhyp")
model = project.version(1).model

cap = cv2.VideoCapture("http://100.81.241.109:8001/video")


def draw_text_with_background(image, text, position, font, font_scale, text_color, bg_color, thickness):
    
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    top_left_bg = (position[0], position[1] - text_height - baseline)
    bottom_right_bg = (position[0] + text_width, position[1] + baseline)

    cv2.rectangle(image, top_left_bg, bottom_right_bg, bg_color, -1)
    cv2.putText(image, text, position, font, font_scale, text_color, thickness)

if not cap.isOpened():
    print("無法打開相機")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取影像幀")
        break
    
    # 影像增強流程
    frame = cv2.convertScaleAbs(frame,  alpha = 1.5, beta = -30) # alpha 為對比度，beta 為亮度
    
    result = model.predict(frame, confidence=5, overlap=85).json()
    predictions = result['predictions']

    if(predictions):
        for prediction in predictions:
            x = prediction['x']
            y = prediction['y']
            width = prediction['width']
            height = prediction['height']
            confidence = prediction['confidence']
            class_name = prediction['class']

            top_left = (int(x - width / 2), int(y - height / 2))
            bottom_right = (int(x + width / 2), int(y + height / 2))
            cv2.rectangle(frame, top_left, bottom_right, (240, 127, 65), 2)
            #cv2.putText(frame, f"{class_name}:{confidence:.2f}", (top_left[0], top_left[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            draw_text_with_background(image=frame, text=f"{class_name}:{confidence:.2f}",position= (top_left[0]-1, top_left[1] - 5), font = cv2.FONT_HERSHEY_SIMPLEX, font_scale = 0.3, text_color = (0, 0, 0), bg_color = (240, 127, 65), thickness=1)
    cv2.imshow('影像辨識', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
