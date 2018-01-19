import torch
import torch.nn as nn
from torch.autograd import Variable

x = Variable(torch.Tensor([5]),requires_grad=True)
#x = torch.nn.Parameter(torch.Tensor([5]))
y = x * x

y.backward()
print(x.grad)
ret = torch.nn.utils.clip_grad_norm([x],3,2)
#print(ret)
print(x.grad)