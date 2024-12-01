from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    
    multiprocessing.freeze_support()

    model = YOLO('models\yolov8n.pt')

    results = model.train(
        data='D:\智慧養殖專題\Code\AI\yaml\data.yaml', # 指定訓練資料集
        imgsz = 640, # 訓練圖片大小 default=640
        epochs = 100, # 訓練次數 default=100
        patience = 15, # 訓練過程中，如果n次沒有進步，則停止訓練 default=10
        batch = 16, # 訓練批次大小 default=16
        project = 'runs/train', # 訓練結果存放位置 default=runs/train
        name = 'exp' # 訓練結果名稱 default=exp
    )