import torch
import json
import torch.nn.functional as F

a = {'a':[1,2,3]}
a_s = json.dumps(a, ensure_ascii=False)
print(type(json.loads(a_s)))
print(a_s)

