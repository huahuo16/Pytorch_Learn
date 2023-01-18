import torch 
import numpy as np 

shape = (4, 4)
rand_tensor = torch.rand(shape)
one_tensor = torch.ones(shape)
zero_tensor = torch.zeros(shape)

print(f"Random_tensor\n {rand_tensor}\n")
print(f"One_tensor\n {one_tensor}\n")
print(f"Zero_tensor\n {zero_tensor}\n")

tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")