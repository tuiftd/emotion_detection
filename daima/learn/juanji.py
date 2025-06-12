#导入库
#导入库
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import time
import os
#设置超参数
input_size = 28*28
hidden_size = 100
output_size = 10
batch_size = 64
learning_rate = 0.005
num_epochs = 10
#设置GPU训练环境
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:',device)

#设置数据预处理、增强方法
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=(-15, 15)),
    #transforms.RandomAffine(degrees=(-15, 15), translate=(0.2, 0.2)),
    transforms.Normalize((0.5,), (0.5,))
])

# 不做归一化
#trans = transforms.ToTensor()

#载入mnist数据
train_data = datasets.MNIST(root='./data',train=True,transform=trans,download=True)
test_data = datasets.MNIST(root='./data',train=False,transform=trans,download=True)

# print(train_data.data)
# print(train_data.targets)

#分批次
train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)

#显示数据
# image,label = next(iter(train_loader))
# image_demo = image[1]
# label_demo = label[1]
#
# image_demo = image_demo.reshape((28,28))
#
# print('图片标签为：',label_demo)
# plt.imshow(image_demo)
# plt.show()

class FileSignalChecker:
    def __init__(self, file_path='stop.flag', cache_interval=5.0):
        self.file_path = file_path
        self.cache_interval = cache_interval
        self.last_check = 0
        self.cached_result = False

    def should_stop(self):
        now = time.time()
        if now - self.last_check > self.cache_interval:
            self.cached_result = os.path.exists(self.file_path)
            self.last_check = now
        return self.cached_result


#搭建神经网络架构
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


#实例化架构
model = NeuralNet(input_size,hidden_size,output_size)
model = model.to(device)

#定义损失和优化器
Loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=5e-6)
# 学习率预热和调度


# Early Stopping参数
patience = 3  # 容忍验证精度不提升的epoch数
min_delta = 0.005  # 最小改进阈值
best_acc = 0.0
best_loss = float('inf')
counter = 0
best_epoch = 0
best_model_weights = None

# 学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',  # 监控验证精度
    factor=0.5,  # 学习率衰减因子
    patience=30,  # 容忍不提升的epoch数
    min_lr=1e-6   # 最小学习率
)
signal_checker = FileSignalChecker(file_path='stop.txt')# 训练结束信号检测器
# 为画图做准备
loss_lst = []
acc_lst = []
val_loss_lst = []

# 标志来判断是否需要加载之前的权重
load_previous_weights = False

# 如果需要加载之前的权重
if load_previous_weights:
    try:
        best_model_weights = torch.load('mnist_model_NG.pth')
        model.load_state_dict(best_model_weights)
        print("Loaded previous weights.")
    except FileNotFoundError:
        print("No previous weights found, starting with initial weights.")

#训练
for epoch in range(num_epochs):
    if signal_checker.should_stop():
        print('Training stopped by signal file')
        break
    # Early Stopping检查
    if counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        model.load_state_dict(best_model_weights)  # 恢复最佳权重
        break
    # 训练阶段
    train_loss = 0
    for batch_index, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        #计算损失
        loss = Loss(outputs,labels)

        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index%100 == 0:
            print('[{}/{}],[{}/{}],loss={:.4f}'.format(epoch+1,num_epochs,batch_index,len(train_loader),loss.item()))
            loss_lst.append(loss.item())
             #测试
            with torch.no_grad():
                model.eval()
                correct_num = 0
                total_num = len(test_data)
                for images, labels in test_loader:
                    images = images.reshape(-1,28*28).to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _,pred = torch.max(outputs,dim=1)
                    correct_num += (pred==labels).sum()
            acc = correct_num/total_num*100
            print('精度={:.4f}%'.format(acc))
            acc_lst.append(acc.item())
            
            # 计算验证损失
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.reshape(-1,28*28).to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    val_loss += Loss(outputs, labels).item()
            val_loss /= len(test_loader)
            val_loss_lst.append(val_loss)
            
            # Early Stopping逻辑
            if acc > best_acc + min_delta:
                best_acc = acc
                best_loss = val_loss
                best_model_weights = model.state_dict()
                best_epoch = epoch
                counter = 0  # 重置计数器
            elif val_loss < best_loss - min_delta:
                best_loss = val_loss
                counter = 0  # 验证损失有改善也重置计数器
            else:
                counter += 1  # 验证指标没有显著提升
            
            # 更新学习率
            scheduler.step(acc)
        model.train()
    print('最佳损失={:.4f}'.format(best_loss))
    print('最佳精度={:.4f}%'.format(best_acc))
    print('最佳模型在第{}轮'.format(best_epoch+1))

# print(loss_lst)
# print(acc_lst)
test_y = []
gp = []

plot_x = np.arange(1,len(loss_lst)+1)
fig, ax1 = plt.subplots()  # subplots一定要带s
ax1.plot(plot_x, loss_lst, c='r')
ax1.set_ylabel('Loss')
ax2 = ax1.twinx()  # twinx将ax1的X轴共用与ax2，这步很重要
ax2.plot(plot_x, acc_lst, c='g')
ax2.set_ylabel('Acc')
plt.show()

conf_matrix = np.zeros((10, 10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(labels)):
            conf_matrix[labels[i]][predicted[i]] += 1

print("Confusion Matrix:")
print(conf_matrix)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(np.arange(10), np.arange(10))
plt.yticks(np.arange(10), np.arange(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# 添加数值和标签
for i in range(10):
    for j in range(10):
        plt.text(j, i, str(int(conf_matrix[i, j])), ha="center", va="center", color="black")

plt.show()

#保存模型权重
torch.save(best_model_weights, 'mnist_model_NC4.pth')


# #拿一张示例图片来测试
# model.eval()
# image,label = next(iter(test_loader))
# image_demo = image[1]
# label_demo = label[1]
#
# with torch.no_grad():
#     output = model(image_demo.reshape(-1,28*28).to(device))
#     _,pred = torch.max(output, dim=1)
# print('预测类别为', pred)
#
# image_demo = image_demo.reshape((28,28))
# print('实际图片标签为：',label_demo)
# plt.imshow(image_demo)
# plt.show()