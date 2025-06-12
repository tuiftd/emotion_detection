import os
import math
import sys
import json
import torch
import cv2 
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from model_v3_fit import mobilenet_v3_large
from model_v3_fit import mobilenet_v3_small
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from cocojoin import  ClassificationDataset
# 全局变量定义
available_scales = [64,64,96,96,96,128,128,128,160]  # 可选尺度
#available_scales = [112]  # 可选尺度
#current_scale = 56  # 当前尺度（替代多进程共享变量）

# class ScaleAwareDataset(torch.utils.data.Dataset):
#     def __init__(self, original_dataset, transform,is_train=True):
#         self.dataset = original_dataset
#         self.transform = transform
#         self.is_train = is_train
#         self.current_scale = 168  # 默认值
        
#     def set_scale(self, scale):
#         self.current_scale = scale
        
#     def __getitem__(self, idx):
#         img, label = self.dataset[idx]
#         return self.transform(img, self.current_scale), label  # 传递current_scale
        
#     def __len__(self):
#         return len(self.dataset)
def letterbox(image, target_size=640):
    # 原始图像尺寸 (注意PIL是width, height顺序)
    w, h = image.size
    # 计算缩放比例
    scale = min(target_size / h, target_size / w)
    # 等比例缩放后的新尺寸
    new_w, new_h = int(w * scale), int(h * scale)
    # 缩放图像 (使用PIL的LANCZOS高质量重采样)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    
    # 创建目标正方形画布 (114是YOLO的填充灰度值)
    canvas = Image.new('RGB', (target_size, target_size), (114, 114, 114))
    
    # 将缩放后的图像居中放置
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    canvas.paste(resized, (left, top))
    
    return canvas
# 训练集预处理（含动态尺度）
class BatchWiseScaleTransform_train:
    def __init__(self,scale_list = [64,64,96,96,96,128,128,128,160] ):
        self.scale_list = scale_list
        self.gray_flage = False
        self.base_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            #transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.7, 1.1)),
            transforms.ColorJitter(0.35, 0.35, 0.35),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.2),
            #transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.3236, std=0.1868) if self.gray_flage else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __call__(self,batch,gray_flage:bool=False):
        scale = random.choice(self.scale_list)
        transformed_batch = []
        self.gray_flage = gray_flage
        for img , label in batch:
            img = transforms.RandomAffine(degrees=25, translate=(0.25, 0.25))(img)
            w, h = img.size
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
            #print(f"Original size: {w}x{h}, Current scale: {scale}")
            resize_crop = transforms.Compose([
                transforms.Resize(int(scale+scale/4)),
                transforms.RandomCrop(scale),
            ])
            img = resize_crop(img)
            img = self.base_transform(img)
            transformed_batch.append((img, label))
        #print(f"Transformed size: {img.size()[0]}x{img.size()[1]}")
        return torch.utils.data.default_collate(transformed_batch)

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

# 定义带预热的余弦退火调度器
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))  
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))  
    return LambdaLR(optimizer, lr_lambda)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # 超参数配置
    batch_size = 28
    epochs = 150
    patience = 30
    best_acc = 0.691
    no_improve_epochs = 0
    warmup_epochs = 15
    train_img_path = r'image_date\Human face emotions_classification\train'
    val_img_path = r'image_date\Human face emotions_classification\valid'
    train_json_path = r'image_date/Human face emotions_classification/train/classification_annotations.json'
    val_json_path = r'image_date\Human face emotions_classification\valid\classification_annotations.json'
    cal_json_path = r'model_pth\mobilenetV3large\class_indices.json'
    # 多进程共享变量初始化
    # # 定义预热阶段
    # warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / warmup_epochs))
    # # 定义余弦退火阶段（预热结束后启动）buzhibuz
    # cosine_scheduler = CosineAnnealingLR(optimizer, T_max= - warmup_epochs)
    # 数据集加载
    train_dataset = ClassificationDataset(train_json_path, train_img_path)
    batch_transform = BatchWiseScaleTransform_train(available_scales)
    val_dataset = ClassificationDataset(val_json_path, val_img_path,transform=BatchWiseScaleTransform_val())
    # train_dataset = datasets.ImageFolder(
    #     root=r'emohunhe\train',
    # )
    # batch_transform = BatchWiseScaleTransform_train(available_scales)
    # # scale_aware_dataset = ScaleAwareDataset(train_dataset, BatchWiseScaleTransform_train(), is_train=True)
    # val_dataset = datasets.ImageFolder(
    #     root=r'emohunhe\val',
    #     transform=BatchWiseScaleTransform_val()
    # )
    #val_dataset = ScaleAwareDataset(val_dataset, BatchWiseScaleTransform_val(), is_train=False)
    # # 类别映射
    class_names = train_dataset.get_class_names()
    #cla_dict = {i: cls_name for i, cls_name in enumerate(class_names)}
    cla_dict =class_names
    with open(cal_json_path, 'w') as f:
        json.dump(cla_dict, f, indent=4)

    # 数据加载器
    nw = min(os.cpu_count(), batch_size if batch_size > 1 else 0, 8)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=batch_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        persistent_workers=True,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Class distribution: {cla_dict}")
    
    # 模型初始化
    #net = mobilenet_v3_large(num_classes=7).to(device)
    net = mobilenet_v3_large(num_classes=8,in_channels=3,min_input_size=112).to(device)
    
    # 加载预训练权重（可选）
    pretrain_path = r'model_pth\mobilenetV3large\emotion_V3_best_model3plus.pth'
    if os.path.exists(pretrain_path):
        net.load_state_dict(torch.load(pretrain_path, map_location=device))
        print(f"Loaded pretrained weights from {pretrain_path}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(net.parameters(), lr=5e-4, weight_decay=5e-3)
    # 定义总训练步数（用于余弦退火）
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs  # 预热步数 = 预热epoch数 × 每epoch步数
    #scheduler = get_cosine_schedule_with_warmup(optimizer,warmup_steps , total_steps)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=7)  # 学习率衰减策略
    # 训练循环
    save_path = r'model_pth\mobilenetV3large\emotion_V3_best_model3plus.pth'
    # print("Train class_to_idx:", train_dataset.class_to_idx)
    # print("Val class_to_idx:", val_dataset.class_to_idx)
    for epoch in range(epochs):
        #global current_scale  # 多进程共享变量
        # 训练阶段
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", file=sys.stdout)
        for images, labels in train_bar:
            #print(f"Batch scale: {images.shape[2]}x{images.shape[3]}")
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #scheduler.step()  # 更新学习率

            
            
            running_loss += loss.item()
            train_bar.set_postfix({"Loss": f"{loss.item():.3f}", "Scale":f"{images.shape[2]}x{images.shape[3]}"})
            #train_bar.set_postfix({"Scale":f"{images.shape[2]}x{images.shape[3]}"})

        # 验证阶段
        net.eval()
        val_acc = 0.0
        val_loss = 0.0
        #val_dataset.set_scale(current_scale)  
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validating", file=sys.stdout)
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                val_bar.set_postfix({ "Scale":f"{images.shape[2]}x{images.shape[3]}"})
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                pred = outputs.argmax(dim=1)
                val_acc += torch.eq(pred, labels).sum().item()
            val_loss /= len(val_loader)
            val_acc /= len(val_dataset)
            print(f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}")
            scheduler.step(val_acc)  # 更新学习率
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(net.state_dict(), save_path)
                no_improve_epochs = 0
                print(f"Best model saved to {save_path}")
            else:
                if epoch>5*warmup_epochs:
                    no_improve_epochs += 1
                print(f"No improve for {no_improve_epochs} epochs")
                if no_improve_epochs >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    print(f"Training finished, best val acc: {best_acc:.3f}")

if __name__ == '__main__':
    main()