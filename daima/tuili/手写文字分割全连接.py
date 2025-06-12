import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2
import time

# 模型和参数设置（与原脚本一致）
#设置超参数
input_size = 28*28
hidden_size = 20
output_size = 10
batch_size = 64
learning_rate = 0.001
num_epochs = 30
white_flat = True
kernel_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
kernel_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 辅助函数（与原脚本一致）
def max_filter(img, kernel_size=3):
    return cv2.dilate(img, kernel=np.ones((kernel_size, kernel_size), np.uint8))

def make_square(img):
    h, w = img.shape
    max_size = max(h, w)
    delta_h = max_size - h
    delta_w = max_size - w
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left
    squared = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return squared

def extract_and_resize_digit(binary_img):
    # contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # filled = np.zeros_like(binary_img)
    
    # if hierarchy is not None:
    #     for i, h in enumerate(hierarchy[0]):
    #         if h[3] !=-1 and h[2] == -1:
    #             cv2.drawContours(filled, contours, i, 0, -1)
    #         else:
    #             cv2.drawContours(filled, contours, i, 255, -1)
    filled = binary_img

    # plt.imshow(filled, cmap='gray')
    # plt.title('filled')
    # plt.show()
    ys, xs = np.where(filled == 255)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((28, 28), dtype=np.uint8)

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    digit_roi = filled[min_y:max_y+1, min_x:max_x+1]
    
    h, w = digit_roi.shape
    scale = 20 / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized_digit = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    #_,resized_digit = cv2.threshold(resized_digit, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    canvas28 = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas28[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_digit
    #canvas28 = max_filter(canvas28, 2)
    # plt.subplot(1, 2, 1)
    # plt.imshow(canvas28, cmap='gray')
    # plt.title('resized_digit')
    

    M = cv2.moments(canvas28)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        shift_x = int(14 - cx)
        shift_y = int(14 - cy)
        M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas28 = cv2.warpAffine(canvas28, M_shift, (28, 28), borderValue=0)
    # plt.subplot(1, 2, 2)
    # plt.imshow(canvas28, cmap='gray')
    # plt.title('shifted')   
    # plt.show()
    return canvas28

#搭建神经网络架构
class NeuralNet(nn.Module):
    def __init__(self,num_input,num_hidden,num_output):
        super(NeuralNet,self).__init__()
        self.fc1 = nn.Linear(num_input,num_hidden)
        self.fc2 = nn.Linear(num_hidden,num_hidden)
        self.fc3 = nn.Linear(num_hidden,num_output)
        self.act = nn.ReLU()

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.fc2(x)
        # x = self.act(x)
        x = self.fc3(x)
        return x

# 加载模型
model = NeuralNet(input_size,hidden_size,output_size)
model.load_state_dict(torch.load('mnist_model_NC2.pth'))
model = model.to(device)
model.eval()

# 图像处理流程
img_path = r"img\real.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('original')
img_copy = img.copy()
img_copy2 = img.copy()
img = img.astype(np.float32)

# 直方图均衡化
img = img.astype(np.float32)*1 #让亮部过曝
img = np.clip(img, 0, 255).astype(np.uint8)
img = cv2.equalizeHist(img)
_, initial_binary = cv2.threshold(img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#获取亮部二值图
contours, _ = cv2.findContours(initial_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    # 找到面积最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)
    mask_255 = np.zeros_like(img, dtype=np.uint8)
    cv2.drawContours(mask_255, [max_contour], -1, 1, thickness=cv2.FILLED)
    
    # 将均衡化结果应用到掩膜区域
    img_copy[mask_255 == 1] = img[mask_255 == 1]
    img_copy = img_copy.astype(np.float32)*2 #让暗部过曝
    img_copy = np.clip(img_copy, 0, 255).astype(np.uint8)
    #img_copy = cv2.equalizeHist(img_copy) #再次强化字符，可能二值化会更好
    # plt.imshow(img_copy, cmap='gray')
    # plt.show()
img_test3 = img_copy
plt.subplot(2, 2, 3)
plt.imshow(img_test3, cmap='gray')
plt.title('预处理，增强文字对比度')
# canny = cv2.Canny(img_test3, 100, 200)
# canny = max_filter(canny, 2)
# plt.imshow(canny, cmap='gray')
# plt.show()
_, binary_image = cv2.threshold(img_test3, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_OPEN, iterations=1)
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_CLOSE, iterations=1)
plt.subplot(2, 2, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('binary_image')
x, y = binary_image.shape[0], binary_image.shape[1]
img_copy = img_copy2

# 查找轮廓并处理
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for contour in contours:
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    if width * height < 1000 or width * height > 1000000:
        continue
        
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(img_copy, [box], 0, (0, 255, 0), 2)
    
    # 提取数字区域
    min_x, max_x = int(min(box[:,0])), int(max(box[:,0]))
    min_y, max_y = int(min(box[:,1])), int(max(box[:,1]))
    roi = binary_image[min_y:max_y, min_x:max_x]
    # roi = img_copy2[min_y:max_y, min_x:max_x]
    # cv2.imshow("digit", roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    digit = extract_and_resize_digit(roi)
    # plt.imshow(digit, cmap='gray')
    # plt.show()
    # 推理并显示结果
    start_time = time.time()
    with torch.no_grad():
        img_tensor = trans(digit)
        img_tensor = img_tensor.reshape(-1, 28*28).to(device)
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        
        # 在框上方显示预测结果
        cv2.putText(img_copy, str(pred.item()), 
                   (min_x + 5, min_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    end_time = time.time()
    print("预测结果：", pred.item(), " 耗时：", end_time - start_time)

# 显示最终结果
plt.subplot(2, 2, 4)
plt.imshow(img_copy, cmap='gray')
plt.title('result')
plt.show()
img_show = cv2.resize(img_copy, (int(y/2), int(x/2)), interpolation=cv2.INTER_AREA)
cv2.imshow("result", img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()