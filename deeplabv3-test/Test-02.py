import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 加载预训练的分割模型（使用新版模型名称）
model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
model.eval()  # 设置为评估模式

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 读取图像
image = Image.open("images/IMG_20220515_092512_2.jpg").convert("RGB")  # 确保是RGB格式

# 应用图像转换
transformed_image = transform(image)

# 预测分割结果
with torch.no_grad():
    output = model(transformed_image.unsqueeze(0))['out'][0]  # 注意获取'out'键

# 处理输出
segmentation = output.argmax(dim=0).detach().cpu().numpy()

# 显示分割结果
plt.imshow(segmentation, cmap='viridis')
plt.axis('off')
plt.colorbar()  # 显示颜色条
plt.show()