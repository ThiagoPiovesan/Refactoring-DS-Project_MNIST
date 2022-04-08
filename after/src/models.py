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
# In this part of the program, we gonna change the forward method,
# because the way it was implemented, it was not optimal.
# So, we gonna use a function composition to make it more efficient, clear
# and not using a lot of variables.
class LinearNet(torch.nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=28 * 28, out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=10),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        return self.network(x)
