from argparse import ArgumentParser, Namespace
from pathlib import Path

import joblib
import nltk
from sklearn_crfsuite import CRF
from termcolor import colored

from src.features import tokens2features
from src.utils import load_color_mapping

nltk.download("punkt")


def parse_arguments() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="If not provided will download weights from the GDrive.",
    )

    parser.add_argument(
        "--sentence",
        type=str,
        required=True,
        help="Path to the .txt file containing text that needs to be parsed.",
    )

    return parser.parse_args()


def print_info() -> None:
    print("Classes have the following colors: ")
    color_mapping = load_color_mapping(fpath=Path("configs", "classes.yaml"))
    result = ""
    for class_name, class_color in color_mapping.items():
        result += "\t- " + colored(class_name, class_color) + "\n"
    print(result)


def gather_result(tokens: list[str], predictions: list[str]) -> str:
    """
    Returns
    -------
    str
        Formatted string to highlight objects.
    """
    color_mapping = load_color_mapping(fpath=Path("configs", "classes.yaml"), obi=True)
    result = "\t"
    for token, prediction in zip(tokens, predictions):
        result += colored(token, color_mapping[prediction]) + " "
    return result


def main(sentence: str, weights: Path | None) -> None:
    print("Loading model...")
    crf_model: CRF = joblib.load(weights)
    print()

    print("Start inferencing...")
    tokens = nltk.tokenize.word_tokenize(sentence)
    predictions = crf_model.predict([tokens2features(tokens=tokens, window_size=2)])
    print()

    print_info()

    print("Inference result: \n")
    print(gather_result(tokens=tokens, predictions=predictions[0]))
    print()


if __name__ == "__main__":
    args = parse_arguments()

    try:
        main(**dict(args._get_kwargs()))
    except KeyboardInterrupt:
        print("User interrupted process.")
