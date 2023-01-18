import torch
x = torch.randn(3, requires_grad=True)
y = x + 2

# print(x) # created by the user -> grad_fn is None
# print(y)
# print(y.grad_fn) 
z = y * y * 3
print(z)
z = z.mean()
print(z)
z.backward()
print(x.grad) # dz/dx
