import torch 
import numpy as np 

tensor = torch.ones(3,4)
mul_tensor = torch.rand(3,4)
tensor[:, 1] = 0 
print(tensor)
print(tensor*mul_tensor)#按位进行乘法(不是矩阵的乘法)
print(tensor @ mul_tensor.T)#进行矩阵的乘法

t1 = torch.cat([tensor,tensor,tensor], dim=1)
print(t1)

# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")