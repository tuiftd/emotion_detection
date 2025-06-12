import os
import cv2
from pathlib import Path

# 配置参数
base_dir = r"C:\Users\gd\Desktop\facecheck"
annotation_file = os.path.join(base_dir, "wider_face_train_bbx_gt.txt")
images_dir = os.path.join(base_dir, "images", "train")
labels_dir = os.path.join(base_dir, "labels", "train")

# 您抽取的类别列表（根据图片中的信息）
selected_categories = [
    "7--Cheering",
    "10--People_Marching",
    "11--Meeting",
    "12--Group",
    "20--Family_Group",
]

# 创建labels目录结构（与images保持一致）
for category in selected_categories:
    os.makedirs(os.path.join(labels_dir, category), exist_ok=True)

def process_annotation_file():
    with open(annotation_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    i = 0
    while i < len(lines):
        # 获取图片路径（格式：category/image_name.jpg）
        img_rel_path = lines[i]
        i += 1
        
        # 提取类别名称
        category = img_rel_path.split('/')[0]
        
        # 跳过未抽取的类别
        if category not in selected_categories:
            # 移动到下一个图片条目（跳过当前图片的所有边界框）
            if i < len(lines):
                try:
                    num_boxes = int(lines[i])
                    i += num_boxes + 1  # 跳过数量行和所有边界框行
                except:
                    i += 1  # 如果格式错误，至少跳过一行
            continue
        
        # 读取边界框数量
        if i >= len(lines):
            break
            
        try:
            num_boxes = int(lines[i])
        except ValueError:
            print(f"格式错误: 无法解析边界框数量 '{lines[i]}' (图片: {img_rel_path})")
            i += 1
            continue
            
        i += 1
        
        # 读取边界框信息
        boxes = []
        for _ in range(num_boxes):
            if i >= len(lines):
                break
            box_info = lines[i].split()
            i += 1
            if len(box_info) >= 4:  # 至少需要x1,y1,w,h
                boxes.append(box_info[:4])  # 只取坐标，忽略属性
        
        # 构建图片完整路径
        img_path = os.path.join(images_dir, img_rel_path)
        
        # 获取图片尺寸
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"图片不存在: {img_path}")
                continue
            img_h, img_w = img.shape[:2]
        except Exception as e:
            print(f"读取图片出错: {img_path}, 错误: {e}")
            continue
        
        # 转换为YOLO格式
        yolo_lines = []
        for box in boxes:
            try:
                x1, y1, w, h = map(float, box)
                
                # 计算归一化坐标
                x_center = (x1 + w/2) / img_w
                y_center = (y1 + h/2) / img_h
                width = w / img_w
                height = h / img_h
                
                # 确保坐标在0-1范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            except Exception as e:
                print(f"转换边界框出错: {box}, 图片: {img_path}, 错误: {e}")
                continue
        
        # 写入标签文件
        if yolo_lines:
            label_filename = os.path.basename(img_rel_path).replace('.jpg', '.txt')
            label_path = os.path.join(labels_dir, category, label_filename)
            
            with open(label_path, 'w') as f:
                f.writelines(yolo_lines)
        else:
            print(f"警告: 没有有效的边界框，跳过: {img_path}")

    print("标签转换完成！")

if __name__ == "__main__":
    process_annotation_file()