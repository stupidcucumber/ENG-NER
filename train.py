from argparse import ArgumentParser, Namespace
from pathlib import Path

import joblib
from sklearn_crfsuite import CRF, metrics

from src.data import DatasetNER


def parse_arguments() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--algorithm",
        type=str,
        default="lbfgs",
        help="Name of one of the algorithms defined in SKlearn CRFSuite.",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum times CRF will iterate over a train dataset.",
    )

    parser.add_argument(
        "--train-data", type=Path, required=True, help="Path to the train dataset."
    )

    parser.add_argument(
        "--test-data", type=Path, required=True, help="Path to the test dataset."
    )

    parser.add_argument(
        "--saving-path",
        type=Path,
        default=Path("output"),
        help="Folder where to put any train info.",
    )

    return parser.parse_args()


def main(
    algorithm: str,
    max_iterations: int,
    train_data: Path,
    test_data: Path,
    saving_path: Path,
) -> None:
    saving_path.mkdir(exist_ok=True)
    model_filename = f"crf_{algorithm}_{max_iterations}.joblib"
    ner_crf_algorithm = CRF(
        algorithm="lbfgs",
        all_possible_transitions=True,
        c1=0,
        c2=0,
        max_iterations=100,
    )

    print("Loading train dataset...")
    train_dataset = DatasetNER(fpath=train_data)
    print()

    print("Start training... Model will be saved to: ", saving_path / model_filename)
    ner_crf_algorithm.fit(train_dataset.features(), train_dataset.labels())
    print()

    print("Loading test dataset...")
    test_dataset = DatasetNER(fpath=test_data)

    y_pred = ner_crf_algorithm.predict(test_dataset.features())
    y_true = test_dataset.labels()

    print("\tACCURACY SCORE: ")
    print(metrics.flat_accuracy_score(y_pred=y_pred, y_true=y_true))

    print("\tF1 SCORE: ")
    print(
        metrics.flat_f1_score(
            y_pred=y_pred,
            y_true=y_true,
            average="weighted",
            labels=sorted(ner_crf_algorithm.classes_),
        )
    )

    print("\tCLASSIFICATION REPORT: ")
    print(
        metrics.flat_classification_report(
            y_pred=y_pred,
            y_true=y_true,
            labels=sorted(ner_crf_algorithm.classes_),
            digits=3,
        )
    )

    print("Saving model...")
    joblib.dump(ner_crf_algorithm, saving_path / model_filename)


if __name__ == "__main__":
    args = parse_arguments()
    try:
        main(**dict(args._get_kwargs()))
    except KeyboardInterrupt:
        print("User interrupted process.")
    else:
        print("Job is done.")
