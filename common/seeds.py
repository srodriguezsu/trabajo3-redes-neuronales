"""Semillas reproducibles compartidas por los tres módulos."""
import os
import random

import numpy as np

SEMILLA_BASE = 42
N_SEMILLAS = 30  # regla de comparación justa: >= 30 corridas por red final


def fijar_semilla(seed: int = SEMILLA_BASE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def semillas_experimento(n: int = N_SEMILLAS) -> list[int]:
    return [SEMILLA_BASE + i for i in range(n)]
