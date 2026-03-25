from pathlib import Path
import tempfile
import shutil
from termcolor import colored
import colorama

colorama.init()

from argparse import ArgumentParser
from datetime import datetime

from holo.prettyFormats import prettyTime
from holo.files import get_unique_name

from paths_cfg import GENERATIONS_DIRECTORY

from CLI.cli_generate import generate_cli


def mass_generate(
    start_file: Path|None,
    model_name: str,
    version_ID: int,
    top_k: int | None,
    temperature: float,
    absolute_gcode: bool,
    relative_gcode: bool,
):
    # global setup
    nbPasses: int = 2
    timeLimites: list[int] = [60*5, 60*10, 60*15]
    # setup the save directory
    saveDir = f"{model_name}_T{round(temperature, 4)}_K{top_k}"
    saveDirPath = GENERATIONS_DIRECTORY.joinpath(saveDir)
    # ensure teh directory exist
    saveDirPath.mkdir(exist_ok=True)
    for passID in range(1, nbPasses+1):
        for time_limite in timeLimites:
            saveFile = get_unique_name(
                saveDirPath, onlyNumbers=True, nbCharacters=None,
                randomChoice=False, allowResize=True, guidlike=False)
            with open(saveDirPath.joinpath("metadatas.txt"), mode="a") as metaFile:
                metaFile.write(f"started: {saveFile} for {time_limite} minutes\n")
            stats = generate_cli(
                start_file=start_file, save_generate=f"{saveDir}/{saveFile}",
                model_name=model_name, version_ID=version_ID,
                time_limit=time_limite, max_tokens=999_999_999,
                top_k=top_k, temperature=temperature,
                absolute_gcode=absolute_gcode, relative_gcode=relative_gcode)
            with open(saveDirPath.joinpath("metadatas.txt"), mode="a") as metaFile:
                metaFile.write(f"finished: {saveFile} with: {stats.value}\n")

if __name__ == "__main__":
    parser = ArgumentParser(description="Generating Model")
    parser.add_argument(
        "--start_file", "--start", type=Path, default=None, 
        help="pour choisir un fichier a utiliser comme debut de generation, " \
            "utilise le début du ficher et non la fin, "\
                "doit etre une svg valide, le </svg> sera retiré" \
                    "(non specifier -> genere un fichier a partir de rien)"
    )
    parser.add_argument(
        "--model_name", "--m", type=str, required=True, help="nom du model a load"
    )
    parser.add_argument(
        "--version_ID",
        "--id",
        type=int,
        required=True,
        help="numero de version du model",
    )
    parser.add_argument(
        "--top_k",
        "--k",
        type=int,
        default=None,
        help="les k meilleurs pour la generation du model (pas specifier -> n'utilise pas de top k)",
    )
    parser.add_argument(
        "--temperature",
        "--temp",
        type=float,
        default=1.0,
        help="temperature pour la generation",
    )
    parser.add_argument(
        "--absolute_gcode",
        "--abs",
        action="store_true",
        help="active le gcode en utilisant les coordonnees absolues",
    )
    parser.add_argument(
        "--relative_gcode",
        "--rel",
        action="store_true",
        help="active le gcode en utilisant les coordonnees absolues",
    )

    args = parser.parse_args()

    tStart = datetime.now()
    mass_generate(
        start_file=args.start_file,
        model_name=args.model_name,
        version_ID=args.version_ID,
        top_k=args.top_k,
        temperature=args.temperature,
        absolute_gcode=args.absolute_gcode,
        relative_gcode=args.relative_gcode,
    )
    print(colored(f"Total time: {prettyTime(datetime.now() - tStart)}", "blue"))