# 导入库
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import random
# 设置超参数
#设置超参数
input_size = 28*28
hidden_size = 500
output_size = 10
batch_size = 512
learning_rate = 0.009
num_epochs = 1000
num_dropout = 0.3
out_channels=16

# 设置GPU训练环境
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置数据预处理、增强方法
trans = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomRotation(degrees=(-15, 15)),
    transforms.Normalize((0.5,), (0.5,))
])

# 不做归一化
#trans = transforms.ToTensor()

# 载入mnist数据
#train_data = datasets.MNIST(root='./data', train=True, transform=trans, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=trans, download=True)

# print(train_data.data)
# print(train_data.targets)

# 分批次
#train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

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

# 加载模型权重
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('mnist_model.pth'))
model.to(device)
model.eval()  # 设置模型为评估模式

# 加载一张示例图片
image, label = next(iter(test_loader))
i=random.randint(0,image.size(0) - 1)
image_demo = image[i]  # 修改这里以确保获取图片
label_demo = label[i]

# 显示原始图片
image_demo_np = image_demo.numpy().reshape((28, 28))
plt.imshow(image_demo_np, cmap='gray')
plt.title('real label: {}'.format(label_demo))
plt.show()
# 将输入图像移动到GPU（如果可用）
#image_demo = image_demo.to(device)
# 进行预测
with torch.no_grad():  # 不需要计算梯度
    image_demo = image_demo.reshape(-1, 28*28).to(device)
    output = model(image_demo)
    top_values, top_indices = torch.topk(output, 10)
    for i in range(10):
        print(f"标签: {top_indices[0][i].item()}, 激活值: {top_values[0][i].item()}")
#     _, predicted = torch.max(output, 1)

# print('predicted label: {}'.format(predicted.item()))

