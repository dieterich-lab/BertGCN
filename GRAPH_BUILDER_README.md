# BertGCN Graph Builder

A robust and configurable graph building system for clinical text classification using document-word heterogeneous graphs.

## Quick Start

### Basic Usage

```bash
# Build a graph with default settings
python -m bertgcn

# Build a graph with custom vocabulary settings
python -m bertgcn --vocab-min-freq 2 --max-vocab-size 100

# Build a graph with custom data split
python -m bertgcn --train-ratio 0.8 --val-ratio 0.1

# Enable verbose logging
python -m bertgcn --verbose
```

### Command Line Options

- `--doclevel TEXT`: Document level (default: "letter")
- `--testunklar`: Enable testunklar mode (flag)
- `--vocab-min-freq INTEGER`: Minimum word frequency threshold (default: 2)
- `--max-vocab-size INTEGER`: Maximum vocabulary size (default: 10000)
- `--train-ratio FLOAT`: Training data ratio (default: 0.7)
- `--val-ratio FLOAT`: Validation data ratio (default: 0.1)
- `--verbose`: Enable verbose logging (flag)

## Programmatic Usage

```python
from bertgcn import build_graph_enhanced

# Build a graph with default settings
result = build_graph_enhanced(doclevel="letter")

# Build a graph with custom parameters
result = build_graph_enhanced(
    doclevel="letter",
    testunklar=False,
    vocab_min_freq=3,
    max_vocab_size=500,
    train_ratio=0.8,
    val_ratio=0.1
)

# Access results
print(f"Graph saved to: {result['graph_dir']}")
print(f"Total nodes: {result['metadata']['total_nodes']}")
print(f"Vocabulary size: {len(result['vocab'])}")
```

## Configuration

### Environment Variables

- `BERTGCN_MODEL_PATH`: Path to the pretrained BERT model (defaults to hardcoded path)

### Default Configuration

```python
{
    "vocab_min_freq": 2,        # Minimum word frequency
    "max_vocab_size": 10000,    # Maximum vocabulary size
    "train_ratio": 0.7,         # Training data ratio
    "val_ratio": 0.1,          # Validation data ratio
    "test_ratio": 0.2,         # Test data ratio (calculated)
}
```

## Output Files

The graph builder creates the following files in `outputs/data/graphs/{graph_name}/`:

### Core Graph Files
- `ind.{graph_name}.adj.npz`: Sparse adjacency matrix (CSR format)
- `ind.{graph_name}.x`: Training features (labels)
- `ind.{graph_name}.vx`: Validation features
- `ind.{graph_name}.tx`: Test features
- `ind.{graph_name}.metadata`: Complete metadata dictionary

### Additional Files
- `ind.{graph_name}.vocab`: Vocabulary mapping (word2id, vocab)
- `ind.{graph_name}.texts`: Processed text data
- `{graph_name}_summary.txt`: Human-readable summary

## Graph Structure

The generated graph is a heterogeneous document-word graph where:

- **Document nodes**: Represent individual clinical documents
- **Word nodes**: Represent vocabulary words
- **Edges**: Connect documents to words based on word occurrence with TF weighting

### Node Indexing
- Document nodes: `0` to `num_docs - 1`
- Word nodes: `num_docs` to `num_docs + num_words - 1`

## Validation and Inspection

### Validate a Graph
```python
from bertgcn.graph_inspector import validate_graph
from pathlib import Path

graph_dir = Path("outputs/data/graphs/medindcls_letter")
is_valid = validate_graph(graph_dir, "medindcls_letter")
```

### Inspect a Graph
```python
from bertgcn.graph_inspector import inspect_graph
from pathlib import Path

graph_dir = Path("outputs/data/graphs/medindcls_letter")
inspect_graph(graph_dir, "medindcls_letter")
```

## Testing

Run the test suite to verify everything works:

```bash
python test_graph_building.py
```

## Architecture

### Key Components

1. **Enhanced Configuration** (`config_enhanced.py`)
   - Flexible model path handling
   - Environment variable support
   - Configurable parameters

2. **Enhanced Graph Builder** (`graph_builder_enhanced.py`)
   - Frequency-based vocabulary filtering
   - TF-weighted edges
   - Comprehensive metadata
   - Error handling and fallbacks

3. **Graph Inspector** (`graph_inspector.py`)
   - Validation utilities
   - Detailed graph statistics
   - Component inspection

4. **Enhanced CLI** (`graph_cli_enhanced.py`)
   - Rich command-line interface
   - Progress reporting
   - Configuration display

### Data Flow

1. **Tokenizer Loading**: Load BERT tokenizer (with fallback)
2. **Dataset Creation**: Generate synthetic clinical data
3. **Vocabulary Building**: Extract and filter vocabulary
4. **Graph Construction**: Build document-word adjacency matrix
5. **Data Splitting**: Split into train/validation/test sets
6. **File Output**: Save all components to disk
7. **Validation**: Verify graph integrity

## Customization

### Custom Dataset

To use your own dataset, modify the `CleanClinicDataset` class in `datasets.py`:

```python
class MyCustomDataset(CleanClinicDataset):
    def _load_data(self):
        # Load your custom data here
        self.texts = your_texts
        self.labels = your_labels
        # ... rest of the implementation
```

### Custom Graph Building

For advanced graph building, extend the `build_graph_enhanced` function:

```python
def my_custom_graph_builder(**kwargs):
    # Your custom logic here
    result = build_graph_enhanced(**kwargs)
    # Additional processing
    return result
```

## Troubleshooting

### Common Issues

1. **Model Path Not Found**
   - Set `BERTGCN_MODEL_PATH` environment variable
   - System will use fallback model automatically

2. **Memory Issues with Large Vocabularies**
   - Reduce `max_vocab_size`
   - Increase `vocab_min_freq`

3. **Invalid Data Splits**
   - Ensure `train_ratio + val_ratio < 1.0`
   - Minimum 1 sample per split

### Debug Mode

Enable verbose logging for detailed debugging:

```bash
python -m bertgcn --verbose
```

## Performance

### Optimization Tips

1. **Vocabulary Size**: Smaller vocabularies = faster processing
2. **Frequency Filtering**: Higher `vocab_min_freq` = smaller graphs
3. **Memory Usage**: Monitor sparse matrix memory consumption

### Benchmarks

- 100 documents, 23 words: ~0.1 seconds
- Memory usage scales with vocabulary size and document length
- Sparse matrices used for memory efficiency

## License

This project is part of the BertGCN framework for clinical text classification.
