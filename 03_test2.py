import torch 

weights = torch.ones(4, requires_grad=True)

# optimizer = torch.optim.SGD(weights, lr=0.01)
# optimizer.step()
# optimizer.zero_grad()

for _ in range(10):
    model_output = (weights*3).sum()
    print(f"model_output:{model_output}\n")
    model_output.backward()
    print(f"weights's grad:{weights.grad}")
    with torch.no_grad():
        weights -= 0.1*weights.grad
    weights.grad.zero_() #在下一次执行时将grad清零

print(weights)