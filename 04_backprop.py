import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute the loss 
y_hat = w * x
loss = (y_hat - y)**2
print(loss)

# backward pass
loss.backward()
print(w.grad)   #通过链式法则求出梯度:dloss/dw

# update weight
## 梯度下降法
for _ in range(100):
    y_hat = w * x
    loss = (y_hat - y)**2
    print(f"loss:{loss}")
    loss.backward()
    with torch.no_grad():
        w -= 0.05*w.grad
    w.grad.zero_()
    
print(w)