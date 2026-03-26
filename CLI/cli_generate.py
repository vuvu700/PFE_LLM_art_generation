from pathlib import Path
from termcolor import colored
import colorama

colorama.init()

from argparse import ArgumentParser
from datetime import datetime

from holo.prettyFormats import prettyTime, SingleLinePrinter
from holo.pointers import Pointer

from paths_cfg import GENERATIONS_DIRECTORY


def get_text(
    start_file: Path,
    absolute_gcode: bool,
    relative_gcode: bool,
) -> str:
    """
    renvois le text du fichier cible dans le format choisit,
    en enlevant l'instruction de fin (</svg> pour les svg, idem pour le gcode)

    args:
        start_file: chemain vers le fichier cible
        absolute_gcode: pour transformer le SVG en gcode absolut
        relative_gcode: pour transformer le SVG en gcode relatif
    """
    import dataset.svg_dataset

    with open(start_file, mode="r") as file:
        content = file.read(-1)
    svgText = dataset.svg_dataset.clean_svg(content)
    if absolute_gcode or relative_gcode:
        text = dataset.svg_dataset.svg_to_gcodes(
            svgText,
            relative=relative_gcode,
        ).removesuffix("G01 X-0")
    else:
        text = svgText.removesuffix("</svg>")
    return text


def generate_cli(
    start_file: Path | None,
    save_generate: str,
    model_name: str,
    version_ID: int,
    time_limit: int | None,
    max_tokens: int | None,
    top_k: int | None,
    temperature: float,
    absolute_gcode: bool,
    relative_gcode: bool,
):
    """
    Boucle pour la generation de svg en ligne de commande.
    args:
        start_file: chemain vers le fichier cible pour commencer la generation
            None -> commence a partir de rien
        save_generate: nom du fichier svg que l'on souhaite generer
        model_name: nom du modele que l'on veut load
        version_ID: version de l'id du model (par rapport au nombre d'epochs entrainer)
        time_limit: temps limite d'entrainement (en secondes)
        max_tokens: max des tokens que l'on ne souhaite pas depasser
        top_k: choix du top_k pour le model
        temperature: temperature de generation
        absolute_gcode: active le gcode en utilisant les coordonnees absolues
        relative_gcode: active le gcode en utilisant les coordonnees absolues
    """
    import torch
    from LLM.model import Model, GenerationStats

    try:
        torch.cuda.empty_cache()
        del model  # type: ignore
        torch.cuda.empty_cache()
    except Exception:
        pass
    model = Model.load(
        model_name, versionID=version_ID, device=torch.device("cuda:0"), compile=False
    )
    model.set_wandb_state(False)
    model.show_infos()

    print(f"trained for {model.nb_epoches_done} epoches")
    for k, v in model.historique.get_all_historique().items():
        print(colored(f"{k}: {v.get((model.nb_epoches_done - 1), None):.4g}", "green"))  # type: ignore

    if start_file is not None:
        print(colored("loading start file", "blue"))
        start = get_text(
            start_file=start_file,
            absolute_gcode=absolute_gcode,
            relative_gcode=relative_gcode,
        )[: model.context_size]
    else:
        start = None

    print(colored("start generating", "blue"))
    statsPtr: Pointer[GenerationStats] = Pointer()
    save_generate_path = GENERATIONS_DIRECTORY / save_generate
    singleLine = SingleLinePrinter(None)
    with open(save_generate_path, "w") as f:
        if start is not None:
            f.write(start)
        else:
            pass  # => empty the file

    for txt in model.generate_flow(
        start=start,
        decode_batch=64,
        temperature=temperature,
        top_k=top_k,
        max_tokens=max_tokens,
        max_time=time_limit,
        statsPtr=statsPtr,
    ):
        with open(save_generate_path, "a") as f:
            f.write(txt)  # append the new text

        singleLine.print(
            f"progress: {statsPtr.value.nb_tokens:_d} tokens generated "
            f"(running for {prettyTime(statsPtr.value.gen_time)})"
        )
    singleLine.newline()

    print(colored(f"finished generating ({statsPtr.value.stop_reason})", "green"))
    print(
        colored(
            f" -> {statsPtr.value.nb_tokens / statsPtr.value.gen_time:.2f} tokens/sec",
            "blue",
        )
    )
    return statsPtr


if __name__ == "__main__":
    parser = ArgumentParser(description="Generating Model")
    parser.add_argument(
        "--start_file",
        "--start",
        type=Path,
        default=None,
        help="pour choisir un fichier a utiliser comme debut de generation, "
        "utilise le début du ficher et non la fin, "
        "doit etre une svg valide, le </svg> sera retiré"
        "(non specifier -> genere un fichier a partir de rien)",
    )
    parser.add_argument(
        "--save_generate",
        "--s",
        type=str,
        required=True,
        help="Non du fichier svg que l'on veut sauvegarder (dans le dossier repository_svg)",
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
        "--time_limit",
        "--t",
        type=int,
        default=None,
        help="Limite de temps en secondes (pas specifier -> pas de limite de temps)",
    )
    parser.add_argument(
        "--max_tokens",
        "--l",
        type=int,
        default=None,
        help="la limite de tokens a generer (pas specifier -> aucune limite)",
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
    generate_cli(
        start_file=args.start_file,
        save_generate=args.save_generate,
        model_name=args.model_name,
        version_ID=args.version_ID,
        time_limit=args.time_limit,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        absolute_gcode=args.absolute_gcode,
        relative_gcode=args.relative_gcode,
    )
    print(colored(f"Total time: {prettyTime(datetime.now() - tStart)}", "blue"))
