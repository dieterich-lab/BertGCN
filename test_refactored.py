#!/usr/bin/env python
# test_refactored.py - Test script to verify refactored module structure

"""
This script imports various modules from the refactored bertgcn package
to verify that everything works correctly.
"""

import inspect
import sys
from pprint import pprint

# Test imports from refactored structure
print("=== Testing Refactored Module Structure ===\n")

try:
    # Import the main package
    import bertgcn

    print(f"Successfully imported bertgcn package from: {bertgcn.__file__}")
    print(f"Version: {bertgcn.__version__}")
    print("\nAvailable components in the public API:")
    pprint(bertgcn.__all__)

    # Import core modules
    from bertgcn.core import get_logger, setup_environment

    print("\n✓ Successfully imported core module")

    # Import config
    from bertgcn.config import DEFAULT_MODEL_PATH, MODEL_PATHS

    print(f"\n✓ Successfully imported config module")
    print(f"  DEFAULT_MODEL_PATH: {DEFAULT_MODEL_PATH}")

    # Import and test params
    from bertgcn.params import parse_args

    print("\n✓ Successfully imported params module")
    # Test that parse_args works without failing
    args = parse_args([])
    print(f"  Args from empty command line: {args}")
    print(f"  Model path from args: {args.model_path}")

    # Import build_graph module
    from bertgcn.build_graph import GraphBuilder

    print("\n✓ Successfully imported build_graph module")
    print(f"  GraphBuilder docstring: {GraphBuilder.__doc__.strip()}")

    # Test logger
    logger = get_logger("test")
    logger.info("✓ Logger is working correctly")

    print("\n=== Refactoring successful! ===")
    print("All modules can be imported and used correctly.")

except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
