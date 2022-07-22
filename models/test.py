import torch
import torch.nn.functional as F

inputs_1 = torch.randn(10, 20, requires_grad=True)
inputs_2 = torch.randn(10, 20, requires_grad=True)

dist = torch.pairwise_distance(inputs_1, inputs_2)
print(dist)

y = torch.zeros(dist.shape[0], dtype=torch.float32)
end = torch.where(dist>6, dist,y)
print(end)
