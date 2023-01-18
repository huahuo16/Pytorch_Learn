import torch 

# 将require_grad转为false三种方法:
# 1. x.requires_grad_(False)
# 2. x.detach()
# 3. with torch.no_grad()

x = torch.randn(3, requires_grad=True)
print(x)



with torch.no_grad():
    y = x + 2
    print(y)