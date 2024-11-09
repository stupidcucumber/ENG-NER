from argparse import ArgumentParser, Namespace
from pathlib import Path

from sklearn_crfsuite import CRF

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
    model_filename = f"crf_{algorithm}_{max_iterations}.model"
    ner_crf_algorithm = CRF(
        algorithm="lbfgs",
        all_possible_transitions=True,
        c1=0,
        c2=0,
        max_iterations=100,
        model_filename=(saving_path / model_filename).as_posix(),
    )

    print("Loading train dataset...")
    train_dataset = DatasetNER(fpath=train_data)

    print("Start training... Model will be saved to: ", saving_path / model_filename)
    ner_crf_algorithm.fit(train_dataset.features(), train_dataset.labels())

    print("Loading test dataset...")
    test_dataset = DatasetNER(fpath=test_data)
    print(
        "TEST ACCURACY RESULT: ",
        ner_crf_algorithm.score(test_dataset.features(), test_dataset.labels()),
    )


if __name__ == "__main__":
    args = parse_arguments()
    try:
        main(**dict(args._get_kwargs()))
    except KeyboardInterrupt:
        print("User interrupted process.")
    else:
        print("Job is done.")
