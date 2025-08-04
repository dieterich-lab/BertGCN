"""
Test script for graph building functionality.
"""

import shutil
import tempfile
from pathlib import Path

from bertgcn.graph_builder_enhanced import build_graph_enhanced
from bertgcn.graph_inspector import inspect_graph, validate_graph


def test_graph_building():
    """Test the complete graph building pipeline."""
    print("🧪 Testing graph building pipeline...")

    # Test basic graph building
    print("\n1. Testing basic graph building...")
    try:
        result = build_graph_enhanced(
            doclevel="letter", testunklar=False, vocab_min_freq=1, max_vocab_size=30
        )
        print("✅ Basic graph building successful")
    except Exception as e:
        print(f"❌ Basic graph building failed: {e}")
        return False

    # Test graph validation
    print("\n2. Testing graph validation...")
    graph_dir = result["graph_dir"]
    graph_name = result["graph_name"]

    if validate_graph(graph_dir, graph_name):
        print("✅ Graph validation successful")
    else:
        print("❌ Graph validation failed")
        return False

    # Test graph inspection
    print("\n3. Testing graph inspection...")
    try:
        inspect_graph(graph_dir, graph_name)
        print("✅ Graph inspection successful")
    except Exception as e:
        print(f"❌ Graph inspection failed: {e}")
        return False

    # Test different configurations
    print("\n4. Testing different configurations...")
    try:
        result2 = build_graph_enhanced(
            doclevel="letter",
            testunklar=True,
            vocab_min_freq=2,
            max_vocab_size=20,
            train_ratio=0.8,
            val_ratio=0.1,
        )
        print("✅ Configuration variations successful")
    except Exception as e:
        print(f"❌ Configuration variations failed: {e}")
        return False

    print("\n🎉 All tests passed!")
    return True


if __name__ == "__main__":
    test_graph_building()
