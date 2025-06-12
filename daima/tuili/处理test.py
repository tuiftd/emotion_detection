import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2
import time
#设置超参数
input_size = 28*28
hidden_size = 100
output_size = 10
batch_size = 512
learning_rate = 0.009
num_epochs = 100
num_dropout = 0.3
out_channels=16
white_flat = True
kernel_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
kernel_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
#设置GPU训练环境
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 设置数据预处理、增强方法
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
def max_filter(img, kernel_size=3):
    """应用最大值滤波"""
    return cv2.dilate(img, 
                     kernel=np.ones((kernel_size, kernel_size), np.uint8))
def make_square(img):
    h, w = img.shape
    max_size = max(h, w)
    
    # 计算需要填充的边距
    delta_h = max_size - h
    delta_w = max_size - w
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left
    
    # 添加黑色边框使图像变为正方形
    squared = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=0)
    return squared

def extract_and_resize_digit(binary_img):
    """
    从一个二值图像中提取字符轮廓（包含孔洞），
    并将其以结构为中心，缩放并居中到28x28图像中。
    """
    # Step 1: 提取所有轮廓（含孔洞）
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary_img)

    # Step 2: 绘制所有轮廓及孔洞
    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            if h[3] !=-1 and h[2] == -1:  # 如果有父轮廓（即当前轮廓是子轮廓）
                cv2.drawContours(filled, contours, i, 0, -1)  # 用0填充子轮廓及其孔洞
            else:  # 如果没有父轮廓（即当前轮廓是父轮廓）
                cv2.drawContours(filled, contours, i, 255, -1)  # 用255填充父轮廓及其孔洞

    # Step 3: 计算非零像素区域的包围盒（避免黑边参与缩放）
    ys, xs = np.where(filled == 255)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((28, 28), dtype=np.uint8)  # 空图像直接返回黑图

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    digit_roi = filled[min_y:max_y+1, min_x:max_x+1]
    # Step 4: 等比缩放到最大边为25
    h, w = digit_roi.shape
    scale = 26 / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized_digit = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    #canvas28 = cv2.copyMakeBorder(resized_digit,1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    # # Step 5: 放置到28x28图像中心
    canvas28 = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas28[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_digit
    # Step 6: 可选 - 质心对齐（将结构中心移动到图像中心）
    M = cv2.moments(canvas28)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        shift_x = int(14 - cx)
        shift_y = int(14 - cy)
        M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas28 = cv2.warpAffine(canvas28, M_shift, (28, 28), borderValue=0)
    return canvas28

class NeuralNet(nn.Module):
    def __init__(self,num_input,num_hidden,num_output):
        super(NeuralNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn_conv = nn.BatchNorm2d(
            out_channels,
            eps=1e-5,       # 默认值，保持数值稳定性
            momentum=0.01,  # 使用较小的动量使统计量更新更平滑
            affine=True     # 保留可学习的缩放和偏移参数
        )  # 添加卷积层后的BatchNorm
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(out_channels*14*14,num_hidden)
        self.bn1 = nn.BatchNorm1d(
            num_hidden,
            eps=1e-5,       # 与卷积层保持一致
            momentum=0.1,  # 全连接层使用默认动量
            affine=True
        )
        self.fc2 = nn.Linear(num_hidden,num_output)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(num_dropout)

    def forward(self,x):
        x = x.view(-1, 1, 28, 28)  # 恢复图像形状 [batch, channel, height, width]
        x = self.conv1(x)
        x = self.bn_conv(x)  # 添加卷积层后的BatchNorm
        x = self.act(x)  # 激活函数在BatchNorm之后
        x = self.pool(x)
        # 全连接层处理
        x = x.view(-1, self.conv1.out_channels*14*14)  # 展平特征图
        x = self.fc1(x)
        x = self.bn1(x)  # 添加BatchNorm
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
#实例化架构
model = NeuralNet(input_size,hidden_size,output_size)
model.load_state_dict(torch.load('mnist_model.pth'))
model = model.to(device)
model.eval()

# 定义与训练时相同的transform（推理时不需要RandomRotation）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#读取图片并预处理
img_path = r"img\real2.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_copy = img.copy()
img_copy2 = img.copy()
#_, initial_binary2 = cv2.threshold(img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 直方图均衡化
img = img.astype(np.float32)*1.5
img = np.clip(img, 0, 255).astype(np.uint8)
img = cv2.equalizeHist(img)
_, initial_binary = cv2.threshold(img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 显示原始图像和直方图均衡化后的图像
cv2.imshow("Original Image", img_copy)
cv2.imshow("Histogram Equalized Image", img)

# 等待按键按下以关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
# 使用initial_binary2找最大轮廓
contours, _ = cv2.findContours(initial_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    # 找到面积最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)
    
    # # 多边形拟合
    # epsilon = 0.00005 * cv2.arcLength(max_contour, True)
    # approx = cv2.approxPolyDP(max_contour, epsilon, True)
    
    # # 创建并填充掩膜
    # mask_255 = np.zeros_like(img, dtype=np.uint8)
    # cv2.fillPoly(mask_255, [approx], 1)
    mask_255 = np.zeros_like(img, dtype=np.uint8)
    cv2.drawContours(mask_255, [max_contour], -1, 1, thickness=cv2.FILLED)
    # 显示最大轮廓
    # 显示掩膜
    plt.imshow(mask_255, cmap='gray')
    plt.title('mask_255')
    plt.show()
    
    # 将均衡化结果应用到掩膜区域
    img_copy[mask_255 == 1] = img[mask_255 == 1]
    img_copy = img_copy.astype(np.float32)*2
    img_copy = np.clip(img_copy, 0, 255).astype(np.uint8)
    #img_copy = cv2.equalizeHist(img_copy)
    plt.subplot(131)
    plt.imshow(img_copy, cmap='gray')
    plt.title('img_copy')
    plt.subplot(132)
    plt.imshow(img_copy2, cmap='gray')
    plt.title('img_copy2')
    plt.subplot(133)
    plt.imshow(img, cmap='gray')
    plt.title('img')
    plt.show()
    # 创建空白图像并填充最大轮廓
    mask_255h = np.zeros_like(mask_255.uint8)
    cv2.drawContours(mask_255h, [max_contour], -1, 255, thickness=cv2.FILLED)
    cv2.imshow("mask_255h", mask_255h)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    mask_255 = (mask_255h == 255).astype(np.uint8)  # 转换回0和1

brightness_255 = np.mean(img[mask_255 == 1])  # 掩膜为1区域的平均亮度
brightness_0 = np.mean(img[mask_255 == 0])    # 掩膜为0区域的平均亮度
print(f"掩膜为1区域的平均亮度: {brightness_255:.2f}")
print(f"掩膜为0区域的平均亮度: {brightness_0:.2f}")


img_bright = img-img_test1
img_bright = np.clip(img_bright, 0, 255).astype(np.uint8)
cv2.imshow("img_bright", img_bright)
cv2.waitKey(0)
cv2.destroyAllWindows()
# img_test1 = np.clip(img_test1, 0, 255).astype(np.uint8)
# cv2.imshow("img_test1", img_test1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# img_test1_inv = cv2.bitwise_not(img_test1)
# img_test2 = img/1.5
# img_test2 = np.clip(img_test2, 0, 255).astype(np.uint8)
img_test3 = (img - img_bright.astype(np.float32)-10)*3
img_test3 = np.clip(img_test3, 0, 255).astype(np.uint8)
cv2.imshow("img_test3", img_test3)
cv2.waitKey(0)
cv2.destroyAllWindows()
# for i in range(10):
#     img_test3 = img_test3-img_test1_inv.astype(np.float32)
img_test3 = np.clip(img_test3, 0, 255).astype(np.uint8)
canny = cv2.Canny(img_test3, 100, 200)
canny = max_filter(canny, 2)
img=img.astype(np.uint8)
# plt.subplot(141)
# plt.imshow(img_test1, cmap='gray')
# plt.title('img_test1')
# plt.subplot(142)
# plt.imshow(img_test2, cmap='gray')
# plt.title('img_test2')
# plt.subplot(143)
# plt.imshow(img, cmap='gray')
# plt.title('img')
# plt.subplot(144)
# plt.imshow(img_test3, cmap='gray')
# plt.title('img_test3')
# plt.show()
# if not white_flat:
#     img = cv2.bitwise_not(img)
#_, binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, binary_image = cv2.threshold(canny, 120, 255, cv2.THRESH_BINARY_INV)
# binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_OPEN, iterations=1)
# binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_CLOSE, iterations=3)
x=binary_image.shape[0]
y=binary_image.shape[1]
binary_show=cv2.resize(binary_image, (int(y/2), int(x/2)), interpolation=cv2.INTER_AREA)
cv2.imshow("binary_image", binary_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
contours_image, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour_image = np.zeros_like(img)
roi_img_list = []
img_copy = img.copy()
for contour in contours_image:
        rect_best = cv2.minAreaRect(contour)
        best_width, best_height = rect_best[1]
        angle = rect_best[2]
        best_width = int(best_width)
        best_height = int(best_height)
        if best_width*best_height < 1000 or best_width*best_height > 1000000:
                continue
        # 获取最小外接矩形的四个顶点坐标
        box = cv2.boxPoints(rect_best)
        box = np.intp(box)
        # 在原始图像上绘制矩形
        cv2.drawContours(img_copy, [box], 0, (0, 255, 0), 2)
        # 将box转换为numpy数组
        box = np.array(box, dtype="int32")
        # 找到最大的x, y和最小的x, y
        max_x = np.max(box[:, 0])
        max_y = np.max(box[:, 1])
        min_x = np.min(box[:, 0])
        min_y = np.min(box[:, 1])
        roi_warped = img[min_y:max_y, min_x:max_x]
        roi_binary = binary_image[min_y:max_y, min_x:max_x]
        mask_moments = np.zeros_like(img)
        cv2.drawContours(mask_moments, [contour], -1, 255, -1)
        roi_moments = mask_moments[min_y:max_y, min_x:max_x]
        #roi_canny = cv2.Canny(roi_warped, 100, 200)
        # resized_image = make_square(roi_canny)
        # resized_image = max_filter(resized_image, 2)
        resized_image = extract_and_resize_digit(roi_binary)
        #resized_image = cv2.resize(resized_image, (28, 28), interpolation=cv2.INTER_AREA)
        
        # 在图像周围填充2像素的黑边
        #resized_image = cv2.copyMakeBorder(resized_image,1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        #blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
        #resized_image = cv2.resize(resized_image, (28, 28), interpolation=cv2.INTER_CUBIC)
        # _, resized_image = cv2.threshold(resized_image, 1, 255, 
        #                         cv2.THRESH_BINARY )
        #resized_image = cv2.anny(resized_image, 100, 200)
        # resized_image = cv2.dilate(resized_image, kernel, iterations=2)
        # resized_image = resized_image*5
        # # #print(resized_image)
        # resized_image = np.clip(resized_image, 0, 255).astype(np.uint8)
       # _, resized_image = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # resized_image = cv2.morphologyEx(resized_image, cv2.MORPH_OPEN, kernel, iterations=1)
        #resized_image = cv2.morphologyEx(resized_image, cv2.MORPH_CLOSE, kernel_CLOSE, iterations=1)

        # # 显示调整后的图像
        # cv2.imshow("Resized Image", resized_image)
        # cv2.waitKey(0)
        # # 关闭所有窗口
        # cv2.destroyAllWindows()
        
        # resized_image = cv2.bitwise_not(resized_image)

        roi_img_list.append(resized_image)
img_show=cv2.resize(img_copy, (int(y/2), int(x/2)), interpolation=cv2.INTER_AREA)
cv2.imshow("img", img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
for img in roi_img_list:
    with torch.no_grad():  # 不需要计算梯度
        # 应用与训练时相同的预处理（除RandomRotation外）
        img_tensor = trans(img)
        img_tensor = img_tensor.reshape(-1, 28*28).to(device)
        output = model(img_tensor)
        top_value, top_index = torch.max(output, 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"预测结果: {top_index.item()}")
        plt.show()
        top_values, top_indices = torch.topk(output, 10)
        # for i in range(10):
        #     print(f"标签: {top_indices[0][i].item()}, 激活值: {top_values[0][i].item()}")
        # print("____________________________________________")
