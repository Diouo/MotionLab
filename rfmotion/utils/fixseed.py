import numpy as np
import torch
import random


def fixseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


SEED = 42
EVALSEED = 0
# Provoc warning: not fully functionnal yet
# torch.set_deterministic(True)
torch.backends.cudnn.benchmark = False

fixseed(SEED)
