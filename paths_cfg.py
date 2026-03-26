"""
permet de configurer les chemains des repertoir
ou l'on retrouve les models, tokenizer et exports
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")


def joinAndEnsure(src: "str|Path", *other: "str|Path") -> Path:
    new = src if isinstance(src, Path) else Path(src)
    for nextDir in other:
        new = new.joinpath(nextDir)
        if new.exists() is False:
            os.mkdir(new)
    assert (
        new.exists()
    ), f"the path: {new.as_posix()!r} don't exist (couldn't be crated)"
    return new


CURRENT_DIRECTORY = Path(__file__).parent
OUR_DATASET_DIRECTORY = CURRENT_DIRECTORY.joinpath("dataset/samples_100/")
OUR_DATASET_DIRECTORY_2 = CURRENT_DIRECTORY.joinpath("dataset/samples_500/")
OUR_DATASET_DIRECTORY_3 = CURRENT_DIRECTORY.joinpath("dataset/samples_1000/")


### dossier pour stocker des fichiers
TOKENIZER_SAVE_DIRECTORY = joinAndEnsure(
    CURRENT_DIRECTORY, os.getenv("TOKENIZER_SAVE_DIRECTORY", default="tokenizer_save")
)
HISTORIQUE_SAVE_DIRECTORY = joinAndEnsure(
    CURRENT_DIRECTORY, os.getenv("HISTORIQUE_SAVE_DIRECTORY", default="historique_save")
)
MODELS_SAVE_DIRECTORY = joinAndEnsure(
    CURRENT_DIRECTORY, os.getenv("MODELS_SAVE_DIRECTORY", default="models_save")
)
GENERATIONS_DIRECTORY = joinAndEnsure(
    CURRENT_DIRECTORY, os.getenv("GENERATIONS_DIRECTORY", default="repository_svg")
)
