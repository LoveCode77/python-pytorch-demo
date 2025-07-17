
import torch
import matplotlib.pyplot as plt

# 准备数据
x = torch.tensor([[1., 2], [2., 3], [3., 4], [4., 5], [5., 6]])
y = torch.tensor([5., 8, 11, 14, 17])

# 定义模型
model = torch.nn.Linear(2, 1)  # 输入维度为 2，输出维度为 1

# 定义损失函数和优化器
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

loss_values = []  # 用于存储每次迭代的损失值

# 训练模型
count=50
for epoch in range(count):
    y_pred = model(x)
    loss = loss_func(y_pred, y.unsqueeze(1))  # 将 y 调整为与 y_pred 相同的形状
    loss_values.append(loss.item())  # 将损失值转换为 Python 数值并存储
    print(f"轮次: {epoch + 1}, 损失: {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 绘制损失值曲线
plt.plot(range(0,count), loss_values)
plt.xlabel('count')
plt.ylabel('loss value')
plt.title('loss picture')

plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
plt.ylim(0,0.5)

plt.show()
