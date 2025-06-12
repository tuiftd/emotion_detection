#导入库
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

#设置超参数
input_size = 3*32*32
hidden_size = 200
output_size = 10
batch_size = 64
learning_rate = 0.001
num_epochs = 5


#设置GPU训练环境
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#设置数据预处理、增强方法
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 不做归一化
# trans = transforms.ToTensor()

#载入mnist数据
train_data = datasets.CIFAR10(root='./data',train=True,transform=trans,download=True)
test_data = datasets.CIFAR10(root='./data',train=False,transform=trans,download=True)

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


#搭建神经网络架构
class NeuralNet(nn.Module):
    def __init__(self,num_input,num_hidden,num_output):
        super(NeuralNet,self).__init__()
        self.fc1 = nn.Linear(num_input,num_hidden)
        self.fc2 = nn.Linear(num_hidden,num_output)
        self.act = nn.ReLU()

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class CNN_1(nn.Module):
    def __init__(self):
        super(CNN_1, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            # nn.Dropout(0.2),
            nn.Linear(2048, 100),
            # nn.ReLU(),
            # nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

#实例化架构
# model = NeuralNet(input_size,hidden_size,output_size)
model = CNN_1()
model = model.to(device)

#定义损失和优化器
Loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# 为画图做准备
loss_lst = []
acc_lst = []

#训练
for epoch in range(num_epochs):
    train_loss = 0
    for batch_index, (images, labels) in enumerate(train_loader):
        # images = images.reshape(-1,32*32*3).to(device)
        images = images.to(device)
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
                    # images = images.reshape(-1,32*32*3).to(device)
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _,pred = torch.max(outputs,dim=1)
                    correct_num += (pred==labels).sum()
            acc = correct_num/total_num*100
            print('精度={:.4f}%'.format(acc))
            acc_lst.append(acc.item())
        model.train()

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
        # images = images.reshape(-1,32*32*3).to(device)
        images = images.to(device)
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



