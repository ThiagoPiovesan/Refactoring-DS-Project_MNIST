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
from pathlib import Path
  
#====================================================================#

def generate_tensorboard_experiment_directory(root: str, parents=True) -> str:
    root = Path(root).resolve()
    child = create_from_missing(root) if not root.exists() else create_from_existing(root)
    child.mkdir(parents=parents)
    return child.as_posix()


def create_from_missing(root):
    return root / '0'


def create_from_existing(root):
    children = [int(c.name) for c in root.glob('*') if (c.is_dir() and c.name.isnumeric())]
    if is_first_experiment(children):
        child = root / '0'
    else:
        child = root / increment_experiment_number(children)
    return child


def is_first_experiment(children: list[int]) -> bool:
    return len(children) == 0


def increment_experiment_number(children: list[int]) -> str:
    return str(max(children) + 1)
