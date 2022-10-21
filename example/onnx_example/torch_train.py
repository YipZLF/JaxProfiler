import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch_alexnet

import torch.profiler as tpr
from torch.profiler import profile, record_function, ProfilerActivity

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
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],record_shapes=True, profile_memory=True, with_stack=True, with_flops= True, with_modules=True) as prof:
        with record_function("Model Train"):
            for i in range(10):
                optimizer.zero_grad()

                pred = model(dummy_input)
                output = loss(pred, target)
                output.backward()

                optimizer.step()
                print("Step {}: {}".format(i+1, output.item()))
    with open("torch_prof.log",'a') as log:
        print(prof.key_averages().table(), file=log)
    prof.export_chrome_trace('./torch_trace.json')


if __name__ == "__main__":
    model = torch_alexnet.Model().cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    train(model, loss, optimizer)
