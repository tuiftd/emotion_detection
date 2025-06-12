import cv2
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO(r"C:\Users\gd\Desktop\facecheck\facecheck\face_v4\weights\best.pt")

# 自定义标签映射（将类别ID映射为中文）
label_map = {
    0: "face",  # 原标签可能是"face"
}

# 加载测试图片
img_path = r"C:\Users\gd\Desktop\AAAdeeplearn\KDEF_result\train\HA\AF01_AF01HAS.JPG"
image = cv2.imread(img_path)
assert image is not None, f"无法读取图片，请检查路径: {img_path}"

# 执行预测
results = model.predict(image, conf=0.5)  # conf为置信度阈值

# 方法1：使用plot()快速绘制（简单修改）
annotated = results[0].plot()  # 先获取基础标注
for box in results[0].boxes:
    x1, y1 = int(box.xyxy[0][0]), int(box.xyxy[0][1])
    cls_id = int(box.cls)
    conf = box.conf.item()
    
    # 覆盖绘制自定义标签（白底黑字）
    custom_text = f"{label_map.get(cls_id, cls_id)} {conf:.2f}"
    cv2.rectangle(annotated, (x1, y1-20), (x1+len(custom_text)*10, y1), (255,255,255), -1)
    cv2.putText(annotated, custom_text, (x1, y1-5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

# 保存结果
output_path = r"C:\Users\gd\Desktop\result.jpg"
cv2.imwrite(output_path, annotated)
print(f"结果已保存至: {output_path}")

# 显示结果（可选）
cv2.imshow("Detection Result", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
