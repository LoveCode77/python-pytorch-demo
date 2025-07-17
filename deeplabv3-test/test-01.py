from PIL import Image
import matplotlib.pyplot as plt
from PIL import  ImageEnhance
from PIL import ImageFilter
# 读取图像
image = Image.open("images/IMG_20220515_092439_1.jpg")

# 显示原始图像
plt.imshow(image)
plt.axis('off')
plt.show()

# 调整图像大小
resized_image = image.resize((200, 300))

# 翻转图像
flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)

# 图像滤波
gaussian_blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))

# 图像增强
enhancer = ImageEnhance.Brightness(image)  # 创建亮度增强器
enhanced_image = enhancer.enhance(2.0)

# 显示调整后的图像
plt.imshow(resized_image)
plt.axis('off')
plt.show()

plt.imshow(flipped_image)
plt.axis('off')
plt.show()

plt.imshow(gaussian_blurred_image)
plt.axis('off')
plt.show()

plt.imshow(enhanced_image)
plt.axis('off')
plt.show()