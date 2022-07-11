import torch

a = torch.tensor([2,3])
sig = torch.nn.Sigmoid()
b1 = sig(a)
print(b1)
b2 = torch.sigmoid(a)
print(b2)
b3 = torch.special.expit(a)
print(b3)
