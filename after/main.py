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
import pathlib   

# Libs importation:
import torch
from src.dataset import create_dataloader
from src.running import Runner, run_epoch

from src.models import LinearNet
from src.utils import generate_tensorboard_experiment_directory

from src.tensorBoard import TensorboardExperiment

#====================================================================#
# Where you are storing what data and which part of your program needs
# what type of data -> determines how the code is going to be structured

# We gonna change this hparams to Constants to make it more clear

# Hyperparameters
EPOCH_COUNT = 20
LR = 5e-5
BATCH_SIZE = 128
LOG_PATH = "./runs"
OPTIMIZER = "Adam"

# Data configuration:
DATA_DIR = "./data"
TEST_DATA = pathlib.Path(f"{DATA_DIR}/t10k-images-idx3-ubyte.gz")
TEST_LABELS = pathlib.Path(f"{DATA_DIR}/t10k-labels-idx1-ubyte.gz")
TRAIN_DATA = pathlib.Path(f"{DATA_DIR}/train-images-idx3-ubyte.gz")
TRAIN_LABELS = pathlib.Path(f"{DATA_DIR}/train-labels-idx1-ubyte.gz")

#====================================================================#

def main():
    # Data
    train_loader = create_dataloader(BATCH_SIZE, TEST_DATA, TEST_LABELS)
    test_loader = create_dataloader(BATCH_SIZE, TRAIN_DATA, TRAIN_LABELS, shuffle=False)

    # Model and Optimizer
    model = LinearNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Create the runners
    train_runner = Runner(train_loader, model, optimizer)
    test_runner = Runner(test_loader, model)

    # Experiment Trackers
    log_dir = generate_tensorboard_experiment_directory(root=LOG_PATH)
    experiment = TensorboardExperiment(log_dir=log_dir)

    for epoch in range(EPOCH_COUNT):
        run_epoch(test_runner, train_runner, experiment, epoch, EPOCH_COUNT)
        
    #--------------------------------------------------------------------#
    experiment.flush()

if __name__ == '__main__':
    main()