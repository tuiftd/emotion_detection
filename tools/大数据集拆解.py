import os
import random
import shutil
import math
import argparse

def distribute_images_randomly(source_folder, num_images, num_folders, output_base="output"):
    """
    随机挑选图片并平均分配到多个文件夹
    
    参数:
        source_folder: 源图片文件夹路径
        num_images: 要挑选的图片总数
        num_folders: 要分配到的文件夹数量
        output_base: 输出文件夹的基础名称(默认为"output")
    """
    # 获取所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    all_images = [f for f in os.listdir(source_folder) 
                 if f.lower().endswith(image_extensions)]
    
    # 检查是否有足够图片
    if len(all_images) < num_images:
        raise ValueError(f"文件夹中只有 {len(all_images)} 张图片，少于要求的 {num_images} 张")
    
    # 随机挑选指定数量的图片
    selected_images = random.sample(all_images, num_images)
    
    # 计算每个文件夹应分配的图片数量
    images_per_folder = math.ceil(num_images / num_folders)
    
    # 创建输出文件夹
    os.makedirs(output_base, exist_ok=True)
    
    # 分配图片到各个文件夹
    for folder_num in range(1, num_folders + 1):
        folder_path = os.path.join(output_base, f"{output_base}_{folder_num}")
        os.makedirs(folder_path, exist_ok=True)
        
        # 计算当前文件夹应分配的图片范围
        start_idx = (folder_num - 1) * images_per_folder
        end_idx = min(folder_num * images_per_folder, num_images)
        
        # 复制图片到当前文件夹
        for img in selected_images[start_idx:end_idx]:
            src_path = os.path.join(source_folder, img)
            dst_path = os.path.join(folder_path, img)
            shutil.copy2(src_path, dst_path)
        
        print(f"文件夹 {folder_path} 已创建，包含 {end_idx - start_idx} 张图片")

if __name__ == "__main__":
    source_folder = "images\dataset"
    num_images = 200
    output_base = r"image\val\background"
    num_folders = 3
    distribute_images_randomly(source_folder = source_folder, num_images = num_images, num_folders = num_folders, output_base = output_base)