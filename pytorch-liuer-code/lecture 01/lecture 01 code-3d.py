import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
x_data=[1.0,2.0,3.0]
y_data=[3.0,5.0,7.0]

def forward(x):
    return x*w+b

def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)*(y_pred-y)

w_list=[]
b_list=[]
mse_list=[]
start_time=time.time()
for w in np.arange(0.0,4.1,0.1):
    print("w=", w)
    w_list.append(w)
    for b in np.arange(0.0,4.1,0.1):
        print("b=",b)
        b_list.append(b)
        l_sum=0
        for x_val,y_val in zip(x_data,y_data):
            y_pred_val=forward(x_val)
            loss_val=loss(x_val,y_val)
            l_sum+=loss_val
            print('\t',x_val,y_val,y_pred_val,loss_val)
        print('MSE=',l_sum/3)
        mse_list.append(l_sum/3)
end_time=time.time()
execution_time = end_time - start_time
print("代码执行时间=",execution_time/1000,"秒")

# 画图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(w_list, b_list, mse_list)

ax.set_xlabel('w_list')
ax.set_ylabel('y_list')
ax.set_zlabel('mse_list')

# plt.plot(w_list,mse_list)
# plt.ylabel('Loss')
# plt.xlabel('w')
plt.show()




