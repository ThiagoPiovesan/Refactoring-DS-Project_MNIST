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

# Libs imporation:
from numbers import Real

#====================================================================#
class Metric:
    values: list[Real]
    running_total: float
    num_updates: float
    average: float

    def __init__(self):
        self.reset()

    def __str__(self):
        return f"Metric(average={self.average:0.4f})"

    def update(self, value: Real, batch_size: int):
        self.values.append(value)
        self.running_total += value * batch_size
        self.num_updates += batch_size
        self.average = self.running_total / self.num_updates

    def reset(self):
        self.values: list[Real] = []
        self.running_total: float = 0.0
        self.num_updates: float = 0.0
        self.average: float = 0.0
