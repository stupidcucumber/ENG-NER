from argparse import ArgumentParser, Namespace
from pathlib import Path


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    
    parser.add_argument(
        "--text", 
        type=Path, 
        required=True, 
        help="Path to the .txt file containing text that needs to be parsed."
    )
    
    return parser.parse_args()


def main(text: Path) -> None:
    pass


if __name__ == "__main__":
    args = parse_arguments()
    
    try:
        main(**dict(args._get_kwargs()))
    except KeyboardInterrupt:
        print("User interrupted process.")
    else:
        print("Job is done!")