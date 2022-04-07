#--------------------------------------------------------------------#
# Project: MNIST Digit Recognizer
# Created by: ArjanCodes - https://youtu.be/ka70COItN40
# Changed by: Thiago Piovesan
# Objective: Learning about code refactoring nad code optimization
#--------------------------------------------------------------------#
# Github repo: https://github.com/ThiagoPiovesan/Refactoring-DS-Project_MNIST
# Github profile: https://github.com/ThiagoPiovesan 
#--------------------------------------------------------------------#
# This is to avoid the dict and tuple type hints erros
from __future__ import annotations  

# Libs importation:
import torch
 
#====================================================================#

class LinearNet(torch.nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()

        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(in_features=28 * 28, out_features=32)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features=32, out_features=10)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
