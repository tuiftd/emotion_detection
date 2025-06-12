import torch
import torch.nn.functional as F

print("CUDA available:", torch.cuda.is_available())
print("cuDNN version:", torch.backends.cudnn.version())

# 测试简单的卷积操作
x = torch.randn(1, 1, 3, 3).cuda()
weight = torch.randn(1, 1, 2, 2).cuda()
output = F.conv2d(x, weight)
print("卷积测试成功，输出形状:", output.shape)