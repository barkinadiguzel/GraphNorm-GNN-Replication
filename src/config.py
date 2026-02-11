import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42

# training basics
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 100

# molecular graph specifics 
MAX_ATOMS = 128
NODE_FEATURE_DIM = 64
EDGE_FEATURE_DIM = 16
