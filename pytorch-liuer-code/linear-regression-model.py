import torch

# 准备数据
x = torch.tensor([[1.], [2.], [3.], [4.], [5.]])
y = torch.tensor([[2.], [4.], [6.], [8.], [10.]])

# 定义模型
model = torch.nn.Linear(1, 1)

# 定义损失函数和优化器
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(1000):
    y_pred = model(x)
    loss = loss_func(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 打印模型的参数（斜率和截距）
print("斜率：", model.weight.item())
print("截距：", model.bias.item())