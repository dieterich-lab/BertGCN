"""CLI entry point for BertGCN."""

import sys
from pathlib import Path


def main():
    """Main CLI dispatcher."""
    if len(sys.argv) < 2:
        print("Usage: bertgcn <command> [args...]")
        print("")
        print("Commands:")
        print("  preprocess    Preprocess the dataset")
        print("  build-graph   Build the document-word graph")
        print("  train         Train the BertGCN model (--dev for 1 epoch test)")
        print("  finetune      Fine-tune BERT")
        print("  predict       Run predictions on test set")
        print("  interpret     Run model interpretations")
        print("")
        print("Use 'bertgcn <command> --help' for more info")
        return

    command = sys.argv[1]
    remaining_args = sys.argv[2:]

    # Import here to avoid circular imports
    if command == "preprocess":
        from .preprocess import preprocess

        preprocess(["override mode:preprocess"])
    elif command == "build-graph":
        from .build_graph import main as build_graph_main

        build_graph_main(remaining_args)
    elif command == "train":
        from .train_gcn import main as train_main

        # Check for --dev flag
        dev_mode = "--dev" in remaining_args
        if dev_mode:
            remaining_args = [arg for arg in remaining_args if arg != "--dev"]
            remaining_args.extend(["dev=true"])

        # Add mode override
        remaining_args.insert(0, "override mode:train_gcn")
        train_main(remaining_args if remaining_args else ["override mode:train_gcn"])
    elif command == "finetune":
        from .train_bert import main as finetune_main

        finetune_main(["override mode:finetune"])
    elif command == "predict":
        from .predict import main as predict_main

        predict_main()
    elif command == "interpret":
        from .interpret import main as interpret_main

        interpret_main(["override mode:train_gcn"])
    else:
        print(f"Unknown command: {command}")
        print("Use 'bertgcn --help' for available commands")
        sys.exit(1)


if __name__ == "__main__":
    main()
