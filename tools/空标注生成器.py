import os
import argparse

def generate_yolo_background_annotations(image_folder, output_folder=None):
    """
    为指定文件夹中的所有图片生成YOLO格式的背景标注文件（空文件）
    
    参数:
        image_folder: 包含图片的文件夹路径
        output_folder: 标注文件输出文件夹（默认与图片同目录）
    """
    # 如果未指定输出文件夹，则使用与图片相同的目录
    if output_folder is None:
        output_folder = image_folder
    else:
        os.makedirs(output_folder, exist_ok=True)
    
    # 支持的图片扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(image_extensions):
            # 构建标注文件名（与图片同名，扩展名改为.txt）
            base_name = os.path.splitext(filename)[0]
            annotation_file = os.path.join(output_folder, f"{base_name}.txt")
            
            # 创建空文件（YOLO格式的背景标注就是空文件）
            with open(annotation_file, 'w') as f:
                pass  # 不写入任何内容
            
            print(f"已创建背景标注文件: {annotation_file}")

if __name__ == "__main__":
    image_folder = r"images\train\background_train"
    output_folder = r"labels\train\background_train"
    generate_yolo_background_annotations(image_folder = image_folder, output_folder = output_folder)