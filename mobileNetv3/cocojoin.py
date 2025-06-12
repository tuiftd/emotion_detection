import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ClassificationDataset(Dataset):
    def __init__(self, json_path, image_dir,transform=None,preload:bool=False):
        """
        简化版分类数据集类
        
        参数:
            json_path: 包含图像信息和类别标签的JSON文件路径
            image_dir: 图像文件所在的目录
        """
        self.transform = transform
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON文件 {json_path} 不存在")
        if not os.path.exists(image_dir):
            raise NotADirectoryError(f"图像目录 {image_dir} 不存在")
        with open(json_path) as f:
            self.annotation_data = json.load(f)
        
        self.image_dir = image_dir
        self.preload = preload
        # 创建类别ID到索引的映射
        self.class_to_idx = {cat['id']: idx for idx, cat in enumerate(self.annotation_data['categories'])}
        
        # 提取图像信息列表
        self.image_infos = self.annotation_data['images']
        # 预加载图像到内存（如果启用）
        self.images = []
        if self.preload:
            for img_info in self.image_infos:
                img_path = os.path.join(image_dir, img_info['file_name'])
                try:
                    image = self._load_image(img_path)
                    self.images.append(image)
                except FileNotFoundError:
                    print(f"警告: 图像 {img_path} 未找到，使用空白RGB替代")
                    self.images.append(Image.new('RGB', (224, 224), (0, 0, 0)))
    def _load_image(self, img_path):
        """统一的图像加载方法"""
        image = Image.open(img_path)
        if image.mode == 'RGBA' or (image.mode == 'P' and 'transparency' in image.info):
            image = image.convert('RGBA')
        elif image.mode == 'P':
            image = image.convert('RGB')
        elif image.mode == 'L':
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        return image
    
    def __len__(self):
        return len(self.image_infos)
    
    def __getitem__(self, idx):
        img_info = self.image_infos[idx]
        
        if self.preload:
            image = self.images[idx]
        else:
            img_path = os.path.join(self.image_dir, img_info['file_name'])
            try:
                image = self._load_image(img_path)
            except FileNotFoundError:
                print(f"警告: 图像 {img_path} 未找到，使用空白RGB替代")
                image = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[img_info['category_id']]
        return image, label
    
    def get_class_names(self):
        """获取类别名称列表，按索引顺序排列"""
        return {cat['id'] :cat['name'] for cat in self.annotation_data['categories']}

# 使用示例
if __name__ == "__main__":
    # 初始化数据集
    dataset = ClassificationDataset(
        json_path='classification_annotations.json',
        image_dir='cropped_images'
    )
    
    # 获取数据加载器
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4
    )
    
    # 示例：获取第一个batch的数据
    images, labels = next(iter(data_loader))
    print(f"Batch图像形状: {images.shape}")  # 应为 [32, 3, 224, 224]
    print(f"Batch标签形状: {labels.shape}")  # 应为 [32]
    
    # 获取类别名称
    class_names = dataset.get_class_names()
    print(f"数据集包含 {len(class_names)} 个类别: {class_names}")