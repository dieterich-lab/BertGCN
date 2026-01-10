from dataclasses import dataclass

import typer


@dataclass
class BertGCNParameters:
    doclevel: str = "letter"
    bertmodel: str = "medbert"
    window_size: int = 20
    batch_size: int = 1000
    bidirectional_tfidf: bool = True
    min_pmi: float = 0.0
    seed: int = 42
    testunklar: bool = False
    legacy_checks: bool = True


def main(
    doclevel: str = typer.Option("letter", help="Document level"),
    bertmodel: str = typer.Option("medbert", help="BERT model"),
    window_size: int = typer.Option(20, help="Window size"),
    batch_size: int = typer.Option(1000, help="Batch size"),
    bidirectional_tfidf: bool = typer.Option(True, help="Bidirectional TF-IDF"),
    min_pmi: float = typer.Option(0.0, help="Minimum PMI"),
    seed: int = typer.Option(42, help="Random seed"),
    testunklar: bool = typer.Option(False, help="Test unclear samples"),
    legacy_checks: bool = typer.Option(
        True, help="Run legacy parity checks to mirror original build_graph.py"
    ),
):
    params = BertGCNParameters(
        doclevel=doclevel,
        bertmodel=bertmodel,
        window_size=window_size,
        batch_size=batch_size,
        bidirectional_tfidf=bidirectional_tfidf,
        min_pmi=min_pmi,
        seed=seed,
        testunklar=testunklar,
        legacy_checks=legacy_checks,
    )
    print("Parameters:")
    for field in params.__dataclass_fields__:
        print(f"  {field}: {getattr(params, field)}")


if __name__ == "__main__":
    typer.run(main)
