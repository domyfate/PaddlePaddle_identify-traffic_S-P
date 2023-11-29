##sad

import torch
from torch import nn

from FunctionModel import FunctionModel

dim = 4
model = FunctionModel(dim=dim)
w = torch.load('model1031.pkl')
model.load_state_dict(w)
for layer in model.parameters():
    layer.requires_grad = False
    #print(layer)
    


orthMatrix = torch.load('orthMatrix.pth')
loss_fn = nn.MSELoss()
attack = nn.Parameter(torch.FloatTensor(dim, dim).uniform_(-1, 1), requires_grad=True)
optimizer = torch.optim.Adam([attack], lr=0.001, betas=(0.9, 0.999), weight_decay=0.0)
epoch = 10000
total_train_step = 0
for _ in range (epoch):
for i in range(10000):
x = torch.FloatTensor(dim, dim).uniform_(-5, 5)
        y = torch.relu(x)
        temp_x = torch.mm(attack.T, torch.mm(x, attack))
        temp_y = torch.mm(attack.T, torch.mm(y, attack))
        temp_x = temp_x.reshape(1, -1)
        temp_y = temp_y.reshape(1, -1)
        optimizer.zero_grad()
        output = model(temp_x)
        loss = loss_fn(output, temp_y)
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{},Loss:{:.4f}".format(total_train_step, loss.item()))
            print("attack:",attack)
            print("----------------")
            print("orrhMatrix:",orthMatrix)
