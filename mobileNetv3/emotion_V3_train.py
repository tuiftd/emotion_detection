import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from model_v3 import mobilenet_v3_large
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# 全局变量定义
available_scales = [112]  # 可选尺度
current_scale = 56  # 当前尺度（替代多进程共享变量）

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

# 训练集预处理（含动态尺度）
class BatchWiseScaleTransform_train:
    def __init__(self,scale_list = [55,55,84,84,84,112,112,112,168,224] ):
        self.scale_list = scale_list
        self.base_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __call__(self,batch):
        scale = random.choice(self.scale_list)
        transformed_batch = []
        for img , label in batch:
            w, h = img.size
            max_size = max(w, h)
            pad_left = (max_size - w) // 2
            pad_top = (max_size - h) // 2
            paddings = (pad_left, pad_top, max_size - w - pad_left, max_size - h - pad_top)
            img= transforms.functional.pad(img, paddings, fill=0)
            #print(f"Original size: {w}x{h}, Current scale: {scale}")
            resize_crop = transforms.Compose([
                transforms.Resize(int(scale+scale/5)),
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
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __call__(self, img):
        w, h = img.size
        max_size = max(w, h)
        pad_left = (max_size - w) // 2
        pad_top = (max_size - h) // 2
        paddings = (pad_left, pad_top, max_size - w - pad_left, max_size - h - pad_top)
        img= transforms.functional.pad(img, paddings, fill=0)
        #print(f"Original size: {w}x{h}, Current scale: {current_scale}")
        resize_crop = transforms.Compose([
            transforms.Resize(84),  # 验证固定使用112
            #transforms.CenterCrop(48-6),
        ])
        img = resize_crop(img)
       # print(f"Transformed size: {img.size()[0]}x{img.size()[1]}")
        return self.base_transform(img)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # 超参数配置
    batch_size = 32
    epochs = 1000
    patience = 30
    best_acc = 0.0
    no_improve_epochs = 0

    # 数据集加载
    train_dataset = datasets.ImageFolder(
        root=r'emo\train',
    )
    batch_transform = BatchWiseScaleTransform_train(available_scales)
    # scale_aware_dataset = ScaleAwareDataset(train_dataset, BatchWiseScaleTransform_train(), is_train=True)
    val_dataset = datasets.ImageFolder(
        root=r'emo\val',
        transform=BatchWiseScaleTransform_val()
    )
    #val_dataset = ScaleAwareDataset(val_dataset, BatchWiseScaleTransform_val(), is_train=False)
    # 类别映射
    class_names = train_dataset.classes
    cla_dict = {i: cls_name for i, cls_name in enumerate(class_names)}
    with open('class_indices.json', 'w') as f:
        json.dump(cla_dict, f, indent=4)

    # 数据加载器
    nw = min(os.cpu_count(), batch_size if batch_size > 1 else 0, 8)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        collate_fn=batch_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Class distribution: {cla_dict}")

    # 模型初始化
    net = mobilenet_v3_large(num_classes=7,input_channels = 1).to(device)
    
    # 加载预训练权重（可选）
    pretrain_path = "emotion_V3gray_best_model_smal.pth"
    if os.path.exists(pretrain_path):
        net.load_state_dict(torch.load(pretrain_path, map_location=device))
        print(f"Loaded pretrained weights from {pretrain_path}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # 训练循环
    save_path = 'emotion_V3gray_best_model_small.pth'
    for epoch in range(epochs):
        global current_scale  # 多进程共享变量
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
                no_improve_epochs += 1
                print(f"No improve for {no_improve_epochs} epochs")
                if no_improve_epochs >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    print(f"Training finished, best val acc: {best_acc:.3f}")

if __name__ == '__main__':
    main()