"""Logging configuration for BertGCN."""

import logging
import sys
import warnings

# Suppress common warnings
warnings.filterwarnings(
    "ignore",
    message="Future Hydra versions will no longer change working directory at job runtime by default.",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message="`evaluation_strategy` is deprecated",
    category=FutureWarning,
)

warnings.filterwarnings(
    "ignore",
    message="Filesystem tracking backend.*is deprecated",
    category=FutureWarning,
)


def setup_logging(log_file=None):
    """Configure logging for clean, real-time output."""
    # Configure root logger
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    # StreamHandler for stdout
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    orig_emit = stream_handler.emit

    def emit_and_flush(record):
        orig_emit(record)
        try:
            stream_handler.flush()
        except Exception:
            pass

    stream_handler.emit = emit_and_flush
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(logging.INFO)

    # File handler if log_file provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set specific loggers to WARNING to reduce noise
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
