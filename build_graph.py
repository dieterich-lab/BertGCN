#!/usr/bin/env python3
"""
Simple script to build graphs for BertGCN.

This script can be run directly without installing the package.
"""

import sys
from pathlib import Path

# Add src to path so we can import without installation
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bertgcn.graph_builder import build_graph_enhanced
from bertgcn.graph_inspector import inspect_graph


def main():
    """Build a graph with some reasonable defaults."""
    print("🚀 BertGCN Graph Builder")
    print("=" * 40)

    try:
        # Build graph with good defaults
        result = build_graph_enhanced(
            doclevel="letter",
            vocab_min_freq=1,
            max_vocab_size=50,
            train_ratio=0.7,
            val_ratio=0.15,
        )

        print(f"\n✅ Graph built successfully!")
        print(f"📁 Location: {result['graph_dir']}")
        print(f"📊 Nodes: {result['metadata']['total_nodes']}")
        print(f"🔗 Edges: {result['metadata']['total_edges']}")
        print(f"📚 Vocabulary: {len(result['vocab'])} words")

        # Show inspection
        print(f"\n📋 Graph Inspection:")
        print("-" * 20)
        inspect_graph(result["graph_dir"], result["graph_name"])

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
