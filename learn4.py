import torch 
import numpy as np

# x = torch.rand(4,4)
# print(x)
# y = x.view(8,2) #进行转换
# print(y)
# z = x.numpy()
# print(type(z))

# a = np.ones(5)
# b = torch.from_numpy(a)
# print(f"b:{type(b)}\n{b}")
# a+=1
# print(a)
# print(b)
# print(b.device)  

x = torch.ones(5, requires_grad=True)
print(x)