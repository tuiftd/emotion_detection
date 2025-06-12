import cv2
import torch
import json
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from model_v3 import mobilenet_v3_large
from model_v3 import mobilenet_v3_small
from PIL import Image

size_dict = {
    "small":64,
    "medium":128
}

# 表情预处理（固定尺度）

class BatchWiseScaleTransform_mobilenetv3:
    def __init__(self):
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __call__(self, img):
        w, h = img.size
        print(f"face_img_size: {w}x{h}")
        # img = np.array(img)
        # # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img= cv2.applyColorMap(img , cv2.COLORMAP_JET)
        # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        max_size = max(w, h)
        pad_left = (max_size - w) // 2
        pad_top = (max_size - h) // 2
        paddings = (pad_left, pad_top, max_size - w - pad_left, max_size - h - pad_top)
        img= transforms.functional.pad(img, paddings,padding_mode='reflect')
        resize_crop = transforms.Compose([
            transforms.Resize(size_dict["small"]),  
            #transforms.CenterCrop(size_dict["small"]),
        ]) if max(w, h) < size_dict["medium"] else  transforms.Compose([
            transforms.Resize(size_dict["medium"]),  
            #transforms.CenterCrop(size_dict["medium"]),
        ])
        img = resize_crop(img)
        return self.base_transform(img)
transform = BatchWiseScaleTransform_mobilenetv3()
# 初始化YOLO人脸检测模型
face_model = YOLO(r"C:\Users\gd\Desktop\facecheck\facecheck\face_v4\weights\best.pt")

# 初始化MobileNetV3表情识别模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model = mobilenet_v3_large(num_classes=8).to(device)

# 加载表情识别模型权重
emotion_model.load_state_dict(torch.load('emotion_V3_best_model3.pth', map_location=device))
emotion_model.eval()

# 加载类别标签
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
emotion_labels = {int(k): v for k, v in class_indices.items()}
print(emotion_labels)

# # 表情识别预处理
# emotion_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# 视频流设置
video_path = 0  # 0=默认摄像头
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "无法打开视频源"

# 实时检测循环
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 执行人脸检测
    results = face_model.predict(frame, conf=0.5, imgsz=512)
    
    # 获取检测结果
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    
    # 对每个检测到的人脸进行表情识别
    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        
        # 提取人脸区域
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:  # 跳过空图像
            continue
        #face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        # 将 OpenCV 图像转换为 PIL 图像
        face_img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))    
        # 预处理人脸图像
        face_input = transform(face_img_pil).unsqueeze(0).to(device)
        
        # 表情识别推理
        with torch.no_grad():
            output = emotion_model(face_input)
            _, predicted = torch.max(output.data, 1)
            emotion_id = predicted.item()
            emotion_id_cope = emotion_id+1
            emotion = emotion_labels.get(emotion_id_cope, "Unknown")
            prob = torch.nn.functional.softmax(output, dim=1)[0][emotion_id].item()
        
        # 绘制边界框和标签
        label = f"{emotion} {prob:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签背景
        cv2.rectangle(frame, (x1, y1-30), (x1+len(label)*10, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # 显示结果
    cv2.imshow("Face Emotion Detection", frame)
    
    # 退出键（ESC或q）
    if cv2.waitKey(100) in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()