"""
Path utilities for BertGCN.

Centralized path management for the project.
"""

from pathlib import Path

# Default paths
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_DATA_DIR = DEFAULT_OUTPUT_DIR / "data"
DEFAULT_MODELS_DIR = DEFAULT_OUTPUT_DIR / "models"
DEFAULT_GRAPHS_DIR = DEFAULT_DATA_DIR / "graphs"
DEFAULT_DATASETS_DIR = DEFAULT_DATA_DIR / "datasets"
DEFAULT_LOGS_DIR = DEFAULT_OUTPUT_DIR / "logs"

# Create directories if they don't exist
for directory in [
    DEFAULT_OUTPUT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_GRAPHS_DIR,
    DEFAULT_DATASETS_DIR,
    DEFAULT_LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


class Paths:
    def __init__(self):
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.data_dir = DEFAULT_DATA_DIR
        self.models_dir = DEFAULT_MODELS_DIR
        self.graphs_dir = DEFAULT_GRAPHS_DIR
        self.datasets_dir = DEFAULT_DATASETS_DIR
        self.logs_dir = DEFAULT_LOGS_DIR

    def get_dataset_path(self, doclevel, model_name, clean=True):
        """Get path for processed dataset."""
        clean_suffix = "_clean" if clean else ""
        return self.datasets_dir / f"{doclevel}_{model_name}{clean_suffix}.pkl"

    def get_graph_path(self, dataset_name, doclevel, testunklar=False):
        """Get path for graph files."""
        graph_dir = self.graphs_dir / f"{dataset_name}"
        graph_dir.mkdir(parents=True, exist_ok=True)
        return graph_dir

    def get_finetuned_model_path(self, doclevel, clean=True):
        """Get path for fine-tuned BERT model."""
        clean_suffix = "_clean" if clean else ""
        return self.models_dir / "finetuned" / f"{doclevel}{clean_suffix}"

    def get_gcn_model_path(self, doclevel, clean=True):
        """Get path for trained GCN model."""
        clean_suffix = "_clean" if clean else ""
        return self.models_dir / "gcn" / f"{doclevel}{clean_suffix}"


# Singleton instance
_paths_instance = None


def get_paths():
    """Get the singleton paths instance."""
    global _paths_instance
    if _paths_instance is None:
        _paths_instance = Paths()
    return _paths_instance
