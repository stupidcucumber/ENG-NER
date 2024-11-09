from pathlib import Path

import pandas as pd

from src.features import numbers2labels, tokens2features


class DatasetNER:
    """Helper class for encapsulating logic of
    working with WikiNER.

    Parameters
    ----------
    fpath : Path
        Path to the file storing data.
    """

    def __init__(self, fpath: Path) -> None:
        self.data = pd.__dict__[f"read_{fpath.suffix[1:]}"](fpath)

    def labels(self) -> list[list[str]]:
        """Returns named labels from the dataset."""
        return [numbers2labels(item) for item in self.data["ner_tags"]]

    def features(self) -> list[list[dict]]:
        """Returns features for each sentence in the dataset."""
        return [tokens2features(sentence) for sentence in self.data["words"]]
