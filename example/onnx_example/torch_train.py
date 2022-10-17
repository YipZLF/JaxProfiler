import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch_alexnet

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(0)

def train(model,loss,optimizer):
    # load data
    dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
    target = torch.randint(low=0, high=1000, size=(10,),device="cuda")
    
    for i in range(10):
        optimizer.zero_grad()

        pred = model(dummy_input)
        output = loss(pred, target)
        output.backward()

        optimizer.step()
        print("Step {}: {}".format(i+1, output.item()))


if __name__ == "__main__":
    model = torch_alexnet.Model().cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    train(model, loss, optimizer)
