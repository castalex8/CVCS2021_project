import os

# Training constants
NUM_EPOCHS = 10
INIT_LR = 1e-3
BS = 128
MARGIN = 1.
MOMENTUM = 0.9
GAMMA = 0.1
STEP_SIZE = 8


def use_lab():
    return bool(os.getenv('USE_LAB'))
