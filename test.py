import torch
import torch.nn as nn

# [2,3,3]
a = torch.tensor([[[1, 2, 3],
                  [1, 2, 3],
                  [1, 2, 3]],

                  [[1, 2, 3],
                   [1, 2, 3],
                   [1, 2, 3]]
                  ],dtype=torch.float32)

# [2, 3]
c = torch.tensor([[1,2,3],
                  [1, 4, 3]]
                  , dtype=torch.float32)

c1 = torch.tensor([[1,2,3],
                  [1, 4, 3]]
                  , dtype=torch.float32)
print(c+c1)
c = torch.unsqueeze(c, dim=2)
print(c)
print(c.size()[1])

d = torch.matmul(a, c)
print(d)

score = d.softmax(dim=1)
print(score)

# [2,3,3]
a = a.transpose(dim0=1,dim1=2)
print(a)

print(torch.matmul(a, score))