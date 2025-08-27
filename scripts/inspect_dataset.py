import argparse
import os

from datasets import load_from_disk
from transformers import AutoTokenizer


def inspect_dataset(dataset_path):
    """
    Loads and inspects a dataset from disk.

    Args:
        dataset_path (str): The path to the dataset directory.
    """
    try:
        # Load dataset
        dataset = load_from_disk(dataset_path)
        print("Dataset loaded successfully!")
        print("\n--- Dataset Info ---")
        print(dataset)
        print("\n--- Features ---")
        print(dataset.features)

        # Load tokenizer from the dataset directory
        tokenizer = AutoTokenizer.from_pretrained(dataset_path)
        print("\n--- Tokenizer loaded successfully from dataset directory ---")

        # Inspect first example
        print("\n--- First example ---")
        example = dataset[0]
        print(example)

        # De-tokenize and compare
        print("\n--- De-tokenization Test ---")
        input_ids = example["input_ids"]
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)

        original_text = example.get("text", "N/A")
        if original_text == "N/A":
            print("Original text not found in dataset.")

        print(f"Original text (first 500 chars): {original_text[:500]}")
        print(f"Decoded text (from input_ids): {decoded_text[:500]}")
        print(f"Length original text: {len(original_text)}")
        print(f"Length decoded text: {len(decoded_text)}")

        print("\nInspection complete.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a Hugging Face dataset.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/processed/tokenized_dataset",
        help="Path to the preprocessed dataset.",
    )
    args = parser.parse_args()
    inspect_dataset(args.dataset_path)
