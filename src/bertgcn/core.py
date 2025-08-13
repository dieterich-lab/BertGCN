"""
Core initialization module for BertGCN project.

This module handles common initialization tasks like:
- Setting random seeds for reproducibility
- Configuring logging
- Suppressing warnings
- Setting environment variables
"""

import logging
import random
import warnings

import numpy as np
import torch

from bertgcn.config import DEFAULT_SEED


def setup_environment(seed: int = DEFAULT_SEED) -> None:
    """
    Set up the environment for BertGCN.

    Args:
        seed: Random seed for reproducibility
    """
    # Suppress warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Name for the logger

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_environment()
    return logger
