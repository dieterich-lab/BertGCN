from unittest.mock import MagicMock

import pytest

# This is our tiny, predictable, fake dataset
MOCK_DATA = """ 1|||diag 1|||anam 1|||risk 1|||A simple graph.|||TypeA|||DrugX|||indication
2|||diag 2|||anam 2|||risk 2|||Another simple document.|||TypeB|||DrugY|||no_indication
3|||diag 3|||anam 3|||risk 3|||A graph is a graph.|||TypeA|||DrugX|||indication
4|||diag 4|||anam 4|||risk 4|||This is the test document.|||TypeC|||DrugZ|||unklar
5|||diag 5|||anam 5|||risk 5|||One more for validation.|||TypeD|||DrugW|||no_indication"""


@pytest.fixture(scope="session")
def mock_csv_path(tmp_path_factory):
    """Creates a temporary CSV file with our mock data for tests to use."""
    p = tmp_path_factory.mktemp("data") / "mock_clinic_data.csv"
    p.write_text(MOCK_DATA)
    return str(p)


def mock_tokenizer_logic(texts_to_tokenize, **kwargs):
    """
    This function mimics the behavior of a real tokenizer in batch mode.
    It takes a list of texts and returns a dictionary where each value
    is a list of the same length.
    """
    num_examples = len(texts_to_tokenize)
    return {
        # For each example, return a dummy list of IDs
        "input_ids": [[1, 2, 3]] * num_examples,
        # For each example, return a dummy attention mask
        "attention_mask": [[1, 1, 1]] * num_examples,
    }


@pytest.fixture(scope="session")
def mock_tokenizer():
    """Mocks the Hugging Face tokenizer to handle batching correctly."""
    tokenizer = MagicMock()
    # Instead of a fixed return_value, we assign a function to side_effect.
    # This function will be called with the same arguments as the tokenizer.
    tokenizer.side_effect = mock_tokenizer_logic
    return tokenizer


@pytest.fixture(scope="module")
def sample_clinic_dataset(mock_csv_path, mock_tokenizer):
    from bertgcn.clinic_datasets import CleanClinicDataset

    dataset = CleanClinicDataset(tokenizer=mock_tokenizer, file_path=mock_csv_path)
    return dataset
