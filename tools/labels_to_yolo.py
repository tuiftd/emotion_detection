import os
from pathlib import Path

# 设置路径
base_dir = r"C:\Users\gd\Desktop\facecheck"
annotation_file = os.path.join(base_dir,"wider_face_train_bbx_gt.txt")
images_dir = os.path.join(base_dir, "images", "train", "WIDER_train", "images")
labels_dir = os.path.join(base_dir, "labels", "train")

# 读取标注文件
with open(annotation_file, 'r') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    # 获取图片路径
    img_rel_path = lines[i].strip()  # 如: 0--Parade/0_Parade_marchingband_1_849.jpg
    i += 1
    
    # 获取图片绝对路径
    img_path = os.path.join(images_dir, img_rel_path)
    
    # 获取图片尺寸 (需要安装OpenCV: pip install opencv-python)
    try:
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            i += int(lines[i]) + 1  # 跳过这个图片的所有标注
            continue
        img_h, img_w = img.shape[:2]
    except:
        img_w, img_h = 1, 1  # 如果无法读取图片，使用默认值
    
    # 获取边界框数量
    num_boxes = int(lines[i].strip())
    i += 1
    
    # 准备YOLO格式标注内容
    yolo_lines = []
    for _ in range(num_boxes):
        box_info = list(map(float, lines[i].strip().split()[:4]))  # 只取前4个值(x1,y1,w,h)
        i += 1
        
        # 转换为YOLO格式 (归一化的中心坐标和宽高)
        x_center = (box_info[0] + box_info[2]/2) / img_w
        y_center = (box_info[1] + box_info[3]/2) / img_h
        width = box_info[2] / img_w
        height = box_info[3] / img_h
        
        # YOLO格式: class x_center y_center width height (class固定为0表示人脸)
        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # 创建对应的标签文件
    label_path = os.path.join(labels_dir, img_rel_path.replace('.jpg', '.txt'))
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    
    with open(label_path, 'w') as f:
        f.writelines(yolo_lines)

print("转换完成！")