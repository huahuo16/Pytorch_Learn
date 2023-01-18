import torch 
x = torch.randn(3, requires_grad=True)
y = x + 2
print(x)
print(y)
print(y.grad_fn) 

z = y * y * 3
#z = z.mean()
print(z)
v = torch.tensor([0.1, 1, 0.01], dtype=torch.float32)
z.backward(v)
print(x.grad) # dz/dx


