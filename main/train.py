import os
from ultralytics import YOLO
import torch.multiprocessing as mp
import torch
def main():
    # 1. 准备路径
    base_dir = r"C:\Users\gd\Desktop\facecheck"
    config_path = os.path.join(base_dir, "data", "widerface.yaml")

    # 2. 检查GPU可用性
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA devices: {torch.cuda.device_count()}")

    # 3. 加载模型
    model = YOLO(r"yolov8s.pt")  # 或 yolov8n.pt/yolov8s.pt 根据显存调整

    # 4. 训练配置（针对WIDER FACE优化）
    training_params = {
        "data": config_path,
        "multi_scale": True,
        "epochs": 150,
        "imgsz":736,
        "batch": 4,              # 显存不足时减小batch
        "workers": 0,            # Windows必须设为0
        "device": "0",          # 指定GPU
        "optimizer": "AdamW",
        "warmup_epochs": 5,            # 热身阶段持续5个epoch
        "warmup_momentum": 0.8,        # 初始动量值（逐步升至0.937）
        "warmup_bias_lr": 0.01,         # 偏置项初始学习率（逐步升至optimizer的lr）
        "cos_lr": True,
        "flipud": 0.1,
        "mixup": 0.1,
        "save_period": 30,
        "project": "facecheck",
        "name": "face_s2",
        "patience": 20,         # 早停
        #"rect": True,          # 启用矩形训练
        "augment": True,        # 启用数据增强
        "label_smoothing": 0.1,
        "fraction":1.0,
        "scale":0.2
    }

    # 5. 开始训练
    results = model.train(**training_params)

    # 6. 验证
    metrics = model.val(date = config_path)
    print(f"mAP@0.5: {metrics.box.map50:.3f}")

if __name__ == "__main__":
    # Windows多进程必须的保护代码
    mp.freeze_support()
    main()