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
from src.running import Runner

from src.dataset import get_train_dataloader, get_test_dataloader
from src.models import LinearNet
from src.utils import generate_tensorboard_experiment_directory

from src.tracking import Stage
from src.tensorBoard import TensorboardExperiment

#====================================================================#
# Where you are storing what data and which part of your program needs
# what type of data -> determines how the code is going to be structured

# Hyperparameters
hparams = {
    'EPOCHS': 20,
    'LR': 5e-5,
    'OPTIMIZER': 'Adam',
    'BATCH_SIZE': 128
}

def main():
    # Data
    train_loader = get_train_dataloader(batch_size=hparams.get('BATCH_SIZE'))
    test_loader = get_test_dataloader(batch_size=hparams.get('BATCH_SIZE'))

    # Model and Optimizer
    model = LinearNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.get('LR'))

    # Create the runners
    train_runner = Runner(train_loader, model, optimizer)
    test_runner = Runner(test_loader, model)

    # Experiment Trackers
    log_dir = generate_tensorboard_experiment_directory(root='./runs')
    experiment = TensorboardExperiment(log_dir=log_dir)

    for epoch in range(hparams.get('EPOCHS')):
        
        experiment.set_stage(Stage.TRAIN)
        train_runner.run("Train batches", experiment)
        
        # Log Train Epoch Metrics
        experiment.add_epoch_metric('accuracy', train_runner.avg_accuracy, epoch)
        
        experiment.set_stage(Stage.VAL)
        test_runner.run("Validation batches", experiment)
        
        # Log Validation Epoch Metrics
        experiment.add_epoch_metric('accuracy', test_runner.avg_accuracy, epoch)
        experiment.add_epoch_confusion_matrix(test_runner.y_true_batches, test_runner.y_pred_batches, epoch)
        
    #--------------------------------------------------------------------#
        # Compute Average Epoch Metrics
        summary = ', '.join([
            f"[Epoch: {epoch + 1}/{hparams.get('EPOCHS')}]",
            f"Test Accuracy: {test_runner.avg_accuracy: 0.4f}",
            f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
        ])
        print('\n' + summary + '\n')
    #--------------------------------------------------------------------#
        # Reset some variables and metrics
        test_runner.reset()
        train_runner.reset()
    #--------------------------------------------------------------------#
    experiment.flush()

if __name__ == '__main__':
    main()