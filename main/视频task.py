import cv2
from ultralytics import YOLO
from torchvision.ops import nms
import torch
import numpy as np
def sliding_window_inference(frame):
    h, w = frame.shape[:2]
    detections = []

    # 滑动窗口分块
    for y in range(0, h, STRIDE):
        for x in range(0, w, STRIDE):
            # 计算窗口区域
            x_end = min(x + WINDOW_SIZE, w)
            y_end = min(y + WINDOW_SIZE, h)
            x_start = max(x_end - WINDOW_SIZE, 0)
            y_start = max(y_end - WINDOW_SIZE, 0)

            patch = frame[y_start:y_end, x_start:x_end]
            results = model.predict(patch, imgsz=WINDOW_SIZE, conf=CONF_THRESH)
            
            # 修改这里：正确处理Results对象
            for result in results:
                boxes = result.boxes  # 获取boxes对象
                if boxes is not None and len(boxes):
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf.item()
                        cls = box.cls.item()
                        if conf >= CONF_THRESH:
                            # 坐标平移回原图
                            detections.append([
                                x1 + x_start, y1 + y_start,
                                x2 + x_start, y2 + y_start,
                                conf, cls
                            ])

    if len(detections) == 0:
        return []

    # NMS合并检测框
    dets = torch.tensor(detections)
    boxes = dets[:, :4]
    scores = dets[:, 4]
    keep = nms(boxes, scores, IOU_THRESH)
    return dets[keep].numpy()

custom_labels = {
    0: 'face'
}
# 参数设置
WINDOW_SIZE = 732
STRIDE = 366  # 滑动步长，窗口之间有重叠
CONF_THRESH = 0.4
IOU_THRESH = 0.5
SKIP_FRAMES = 2
# 加载最佳模型（选择您的版本，例如face_v17）
model = YOLO(r"facecheck\face_s2\weights\best.pt")

# 视频流来源（可选摄像头/视频文件/RTSP流）
video_path = "vision\DJI_20250506100839_0780_D.MP4"  # 0=默认摄像头，或替换为视频文件路径

# 初始化视频捕获
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "无法打开视频源"
frame_count = 0
last_detections = []
# 实时检测循环
while cap.isOpened():
    ret, frame = cap.read()
    X, Y, _ = frame.shape
    scl = 732 / min(X, Y)
    frame = cv2.resize(frame, (int( Y* scl), int( X * scl)))
    print(frame.shape)
    if not ret:
        break
    frame_count += 1
    # 跳帧检测逻辑
    if frame_count % (SKIP_FRAMES + 1) == 0 or frame_count == 1:
        # 关键帧：进行完整检测
        last_detections = sliding_window_inference(frame)
    else:
        # 非关键帧：使用上一帧的检测结果
        pass  # 直接使用last_detections
    #detections = sliding_window_inference(frame)

    # 画检测框
    for x1, y1, x2, y2, conf, cls in last_detections:
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('YOLO Sliding Window Detection', frame)
    if cv2.waitKey(10) == 27:  # 按 ESC 退出
        break

cap.release()
cv2.destroyAllWindows()