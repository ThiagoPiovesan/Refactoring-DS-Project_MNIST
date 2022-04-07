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

# Libs Imporation:
import numpy as np
from numbers import Real
from typing import Protocol, Union
from enum import Enum, auto            

#====================================================================#
# First step: Changing the dataclass(frozen) to an Enum 

#====================================================================#
# Stage Class:
class Stage(Enum):
    TRAIN = auto()
    TEST = auto()
    VAL = auto()

#====================================================================#

# Experiment Tracker Class:
class ExperimentTracker(Protocol):

    def add_batch_metric(self, name: str, value: Real, step: int):
        """Implements logging a batch-level metric."""

    def add_epoch_metric(self, name: str, value: Real, step: int):
        """Implements logging a epoch-level metric."""

    def add_epoch_confusion_matrix(self, y_true: np.array, y_pred: np.array, step: int):
        """Implements logging a confusion matrix at epoch-level."""

    def add_hparams(self, hparams: dict[str, Union[str, Real]], metrics: dict[str, Real]):
        """Implements logging hyperparameters."""

    def set_stage(self, stage: Stage):
        """Implements setting the stage of the experiment."""      

    def flush(self):
        """Implements flushing the metrics to disk."""    
    