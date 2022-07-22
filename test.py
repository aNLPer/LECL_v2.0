import torch

a = torch.tensor([[1,2,3],[1,2,3]], dtype=torch.float32)
b = torch.tensor([[1],[-torch.inf]], dtype=torch.float32)

print(torch.exp(torch.tensor(float("-inf"))))
print(torch.tensor(float("-inf")))

