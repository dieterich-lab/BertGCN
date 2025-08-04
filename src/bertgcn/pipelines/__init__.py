"""BertGCN pipelines package."""

from .inference import InferencePipeline
from .training import MLOpsTrainingPipeline

__all__ = ["MLOpsTrainingPipeline", "InferencePipeline"]
