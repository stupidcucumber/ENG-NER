from pathlib import Path

import gdown
import pyaml


def load_class_mapping(fpath: Path) -> dict[int, str]:
    """Loads class indices to their names.

    Parameters
    ----------
    fpath : Path
        Path to the configuration file.

    Returns
    -------
    dict[int, str]
        Mapping of the class ids to their names.
    """
    _config: dict = pyaml.yaml.safe_load(fpath.read_text())
    return {class_id: value["name"] for class_id, value in _config.items()}


def load_color_mapping(fpath: Path, obi: bool = False) -> dict[int, str]:
    """Loads mapping between classes and the color.

    Parameters
    ----------
    fpath : Path
        Path to the yaml configuration file.
    obi : bool, default=False
        Whether to include BEGIN and IN tokens, e.g. B-ORG, I-ORG.

    Returns
    -------
    dict[int, str]
        ...
    """
    _config: dict = pyaml.yaml.safe_load(fpath.read_text())
    _mapping = {value["name"]: value["color"] for value in _config.values()}

    if not obi:
        return _mapping

    result = {}

    for key, value in _mapping.items():
        if key == "NON-ENTITY":
            result[key] = value
        else:
            result[f"B-{key}"] = value
            result[f"I-{key}"] = value

    return result


def download_default_weights() -> Path:
    """Downloads default weights of the CRF model from the google drive.

    Returns
    -------
    Path
        Path to the default weights on a local machine.
    """
    url = "https://drive.google.com/uc?id=1ZNBjtGVFe2kHaPO2DfXLKNr7It00VMdN"
    output = Path("default.joblib")
    if not output.exists():
        print("Downloading default model weights...")
        gdown.download(url, output.as_posix())
    return output
