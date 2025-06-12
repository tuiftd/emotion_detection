import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model_v3_fit import mobilenet_v3_large
import numpy as np
import cv2
from PIL import Image
from cocojoin import  ClassificationDataset
# 验证集预处理（固定尺度）
class BatchWiseScaleTransform_val:
    def __init__(self):
        self.gray_flage = False
        self.base_transform = transforms.Compose([
            #transforms.Grayscale(num_output_channels=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=0.3236, std=0.1868) if self.gray_flage else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __call__(self, img, gray_flage:bool=False):
        w, h = img.size
        self.gray_flage = gray_flage
        if gray_flage:
                img = np.array(img)
                if len(img.shape) == 3:
                    if img.shape[2] == 3:  # RGB -> 灰度
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    elif img.shape[2] == 4:  # RGBA -> 灰度
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                    else:  # 单通道或多通道（非RGB）
                        img = img.mean(axis=-1).astype(np.uint8)  # 取均值或保留第一通道
                else:  # 直接是 (H, W) 的灰度图
                    pass
                img = Image.fromarray(img)
        # if img.shape[0] == 3:
        #     img = img.mean(dim=0, keepdim=True)
        max_size = max(w, h)
        pad_left = (max_size - w) // 2
        pad_top = (max_size - h) // 2
        paddings = (pad_left, pad_top, max_size - w - pad_left, max_size - h - pad_top)
        img= transforms.functional.pad(img, paddings,padding_mode = 'reflect')
        #print(f"Original size: {w}x{h}, Current scale: {current_scale}")
        resize_crop = transforms.Compose([
            transforms.Resize(int(96+15)),  # 验证固定使用112
            transforms.CenterCrop(96),
        ])
        img = resize_crop(img)
       # print(f"Transformed size: {img.size()[0]}x{img.size()[1]}")
        return self.base_transform(img)

# ---------- 1. 加载类名映射 ----------
train_img_path = r'image_date\Human face emotions_classification\train'
val_img_path = r'image_date\Human face emotions_classification\valid'
train_json_path = r'image_date/Human face emotions_classification/train/classification_annotations.json'
val_json_path = r'image_date\Human face emotions_classification\valid\classification_annotations.json'
cal_json_path = r'model_pth\mobilenetV3large\class_indices.json'
class_path = "model_pth\mobilenetV3large\class_indices.json"
with open(class_path, "r") as f:
    class_dict = json.load(f)  # e.g., {"3": "cat", "5": "dog", "7": "car"}


test_dataset = ClassificationDataset(val_json_path, val_img_path,transform=BatchWiseScaleTransform_val())
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# 排序后的原始ID列表和类别名列表
class_dict = test_dataset.get_class_names()
sorted_ids = sorted(class_dict.keys(), key=lambda x: int(x))
original_ids = [int(k) for k in sorted_ids]
class_names = [class_dict[i] for i in sorted_ids]

# 原始ID → 连续索引（用于混淆矩阵）
id_to_index = {cid: idx for idx, cid in enumerate(original_ids)}

# ---------- 3. 加载模型 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mobilenet_v3_large(num_classes=8,in_channels=3,min_input_size=112).to(device)
pretrain_path = r'model_pth\mobilenetV3large\emotion_V3_best_model3.pth'
if os.path.exists(pretrain_path):
    model.load_state_dict(torch.load(pretrain_path, map_location=device))
    print(f"Loaded pretrained weights from {pretrain_path}")
model.eval()

# ---------- 4. 推理并收集标签 ----------
y_true_ids = []
y_pred_ids = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        labels = labels.tolist()

        index_to_id = {idx: int(k) for idx, k in enumerate(sorted_ids)}  # 0 → 1，1 → 2 ...

        true_ids = [index_to_id[label] for label in labels]
        pred_ids = [index_to_id[pred] for pred in preds]
        y_true_ids.extend(true_ids)
        y_pred_ids.extend(pred_ids)

# ---------- 5. 构建混淆矩阵 ----------
y_true_index = [id_to_index[i] for i in y_true_ids]
y_pred_index = [id_to_index[i] for i in y_pred_ids]

def compute_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

cm = compute_confusion_matrix(y_true_index, y_pred_index, num_classes=len(class_names))

recall_per_class = {}
misclassification_rate_per_class = {}

for i in range(len(class_names)):
    TP = cm[i, i]  # 真正例
    FN = cm[i, :].sum() - TP  # 假负例
    FP = cm[:, i].sum() - TP  # 假正例
    TN = cm.sum() - (TP + FN + FP)  # 真负例

    # 召回率（Recall）
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    recall_per_class[class_names[i]] = recall

    # 误识别率（Misclassification Rate）
    misclassification_rate = FP / (TP + FN) if (TP + FN) > 0 else 0
    misclassification_rate_per_class[class_names[i]] = misclassification_rate

# 输出误识别率和召回率
print("Recall per class:")
for class_name, recall in recall_per_class.items():
    print(f"{class_name}: {recall:.4f}")

print("\nMisclassification Rate per class:")
for class_name, rate in misclassification_rate_per_class.items():
    print(f"{class_name}: {rate:.4f}")
# ---------- 6. 绘制混淆矩阵 ----------
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(cm, class_names)
