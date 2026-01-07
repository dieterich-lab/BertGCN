"""Custom Hydra resolvers for the BertGCN project.

This module must be imported BEFORE any @hydra.main decorated functions are evaluated
so that the resolvers are available during config interpolation.
"""

from pathlib import Path

from omegaconf import OmegaConf


def _basename(path_str: str) -> str:
    """Return the final path component (similar to UNIX basename)."""
    try:
        return Path(path_str).name
    except Exception:
        # Fallback: return unchanged
        return path_str


# Register resolvers idempotently
def _register():
    try:
        existing = getattr(OmegaConf, "get_resolver_names", lambda: set())()
        if "basename" not in existing:
            # Support both older and newer API names
            register_fn = getattr(
                OmegaConf,
                "register_new_resolver",
                getattr(OmegaConf, "register_resolver", None),
            )
            if register_fn is None:
                raise RuntimeError("No resolver registration function available")
            register_fn("basename", _basename)
    except Exception:
        pass


_register()
