### 梯度下降法进行训练
import torch
import torch.nn as nn

# Here we replace the manually computed gradient with autograd

# Linear regression
# f = w * x * x 

# here : f = 1.5 * x * x
# 进行学习训练的数据
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[4], [8], [12], [16]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32) 
n_samples, n_features = X.shape

input_size = n_features
output_size = n_features 

# model = nn.Linear(input_size, output_size) 


"""定义自己的modul"""
class LinearRegression(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #define layers 
        self.lin = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.lin(x)
    
model = LinearRegression(input_size, output_size )

# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# # model output
# def forward(x):
#     return w * x * x

# loss = MSE
# def loss(y, y_pred):
#     return ((y_pred - y)**2).mean()

loss = nn.MSELoss()


print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 100

#optimizer = torch.optim.SGD([w], lr=learning_rate)   #随机梯度下降法
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate )

for epoch in range(n_iters):
    # predict = forward pass
    #y_pred = forward(X)
    y_pred = model(X) 

    # loss
    l = loss(Y, y_pred)

    # calculate gradients = backward pass
    l.backward()

    # update weights
    #w.data = w.data - learning_rate * w.grad
    # with torch.no_grad():
    #     w -= learning_rate * w.grad
    optimizer.step()
    
    
    # zero the gradients after updating
    # w.grad.zero_()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l.item():.8f}')

print(f'Prediction after training: f(5) = {model(X_test ).item():.3f}')
