import torch

# 准备数据
x = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],dtype=torch.float)
y = torch.tensor([5, 8, 11, 14, 17],dtype=torch.float)

# 定义模型
model = torch.nn.Linear(2, 1)  # 输入维度为 2，输出维度为 1

# 定义损失函数和优化器
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(1000):
    y_pred = model(x)
    loss = loss_func(y_pred, y.unsqueeze(1))  # 将 y 调整为与 y_pred 相同的形状
    print(f"轮次: {epoch + 1}, 损失: {loss}")
    y.unsqueeze(1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 打印模型的参数（权重和偏置）
# print("权重：", model.weight)
# print("偏置：", model.bias)
torch.save(model.state_dict(),'model_weights.pth')
# 进行预测
new_x = torch.tensor([[7, 8]],dtype=torch.float)
prediction = model(new_x)
print("预测值：", prediction)