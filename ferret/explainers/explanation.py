from dataclasses import dataclass, field

import numpy as np


@dataclass
class Explanation:
    """Generic class to represent an Explanation"""

    text: str
    tokens: str
    scores: np.array
    explainer: str
    target: int
    base_values: np.array = None
    pred: np.array = None
    # HACK
    rationale: np.array = None


@dataclass
class ExplanationWithRationale(Explanation):
    """Specific explanation to contain the gold rationale"""

    rationale: np.array
