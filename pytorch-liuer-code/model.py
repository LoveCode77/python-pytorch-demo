import torch

loaded_model = torch.nn.Linear(2, 1)
loaded_model.load_state_dict(torch.load('model_weights.pth'))

# 进行预测
new_x = torch.tensor([[98, 99]],dtype=torch.float)
prediction = loaded_model(new_x)
print("预测值：", prediction)