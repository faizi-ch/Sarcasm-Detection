import os

CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5"
# CUDA_VISIBLE_DEVICES = "0"

random_state = 42

data_file = "data/train-balanced-sarcasm.csv"

test_size = 0.25


num_classes = 4  # Number of classes including background
num_labels = 3  # Number of classes excluding background
learning_rate = 1e-5
lr_backbone = 1e-5
weight_decay = 1e-4
num_epochs = 10
batch_size = 128
num_workers = 80  # Number of workers for the dataloaders

threshold = (
    0.5  # Score threshold to convert model's output probabilities to binary predictions
)

EARLY_STOPPING = 3
