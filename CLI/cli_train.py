from pathlib import Path
from termcolor import colored
import colorama

colorama.init()

from argparse import ArgumentParser
from datetime import datetime

from .presets import PRESETS
from holo.prettyFormats import prettyTime
from paths_cfg import TOKENIZER_SAVE_DIRECTORY


def train_cli(
    dataset_path: Path,
    save_name: str,
    preset: str,
    max_epochs: int,
    time_limit: int,
    tokenizer_name: str,
    absolute_gcode: bool,
    relative_gcode: bool,
    versionID: int,
    wandb: bool,
):
    """
    Boucle pour generer l'entrainement en ligne de commande.

    args:
        dataset_path: le chemin du dataset
        save_name: nom de sauvegarde du modele (peut etre un nom existant)
        versionID: pour selectioner une version specifique du model
            dont on reprendra l'entrainement
        preset: les parametre du LLM predefis pour l'entrainement de chaque modele.
            donnés dans le fichier CLI/presets.py
        max_epochs: nombre maximum d'epochs sur lequel on veut s'entrainer.
        time_limit: temps limite d'entrainement (en minutes)
            si une epoche est commencée elle n'est pas interompue par la limite de temps
        tokenizer_name: nom du tokenizer que l'on souhaite utiliser (dans le dossier tokenizer_save)
            si il n'existe pas, cela vas le creer automatiquement
        absolute_gcode: active le gcode en utilisant les coordonnees absolues
        relative_gcode: active le gcode en utilisant les coordonnees absolues
        wandb: permet d'activer / desactiver l'utilisation de wandb pour monitorer les entrainements
    """
    use_gcode: bool = absolute_gcode or relative_gcode
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    print(colored("dataset_path valid", "green"))

    tokenizer_path = TOKENIZER_SAVE_DIRECTORY / tokenizer_name

    if not tokenizer_path.exists():
        print(colored("tokenizer not found", "red"))
        answer = input("train tokenizer? (y/n): ")

        if answer.lower() == "y":
            print(colored("loading dataset, this can take some time", "green"))
            dataset = svg_dataset.SVGDataset(
                dataset_path,
                context_size=4096,
                use_gcode=use_gcode,
                use_relative_gcode=relative_gcode,
            )

            print(colored("training tokenizer, this can take some time", "green"))
            tokenizer = tokenizerLib.Tokenizer.train_from_iterator(
                (svg.txt for svg in dataset.samples),
                vocab_size=1024,
                special_tokens=tokenizerLib.SPECIAL_TOKENS,
            )

            tokenizer.save(tokenizer_path)
            del dataset
            gc.collect()
            torch.cuda.empty_cache()

        else:
            print(colored("fail to import tokenizer", "red"))
            return

    else:
        print(colored("loading tokenizer", "green"))
        tokenizer = tokenizerLib.Tokenizer.load(tokenizer_path)  # type: ignore

    preset_config = PRESETS[preset]

    print(colored("loading dataset, this can take some time", "green"))
    dataset = svg_dataset.SVGDataset(
        dataset_path,
        context_size=preset_config["context_size"],
        tokenizer=tokenizer.encode,
        decoder=tokenizer.decode,
        use_gcode=use_gcode,
        use_relative_gcode=relative_gcode,
    )

    if versionID != None:
        model = Model.load(
            save_name, versionID=versionID, device=torch.device("cuda:0"), compile=True
        )
    else:
        model = Model(
            save_name=save_name,
            depth=preset_config["depth"],
            head_dim=preset_config["head_dim"],
            context_size=preset_config["context_size"],
            nb_heads_mult=preset_config["nb_heads_mult"],
            tokenizer=tokenizer_path,
            device="cuda",
        )
    model.set_wandb_state(wandb)
    model.show_infos()

    print(colored("start training", "green"))

    model.train(
        dataset=dataset,
        batch_size=8,
        nbEpoches=max_epochs,
        timeLimite=time_limit,
        verbose=Verbose.liveProgress,
    )

    print(colored("end training", "green"))
    print(colored("metrics", "green"))

    for name, values in model.historique.get_all_historique().items():
        try:
            last_value = list(values.values())[-1]
            print(colored(f"{name}: {last_value:.4g}", "green"))
        except Exception:
            pass

    print(colored(f"model save name: {save_name}", "blue"))


if __name__ == "__main__":
    parser = ArgumentParser(description="Training Model")
    parser.add_argument("--dataset_path", "--d", type=Path, help="le chemin du dataset")
    parser.add_argument(
        "--save_name",
        "--s",
        type=str,
        help="nom de sauvegarde du modele (peut etre un nom existant)",
    )
    parser.add_argument(
        "--versionID",
        "--v",
        type=int,
        required=False,
        default=None,
        help="pour selectioner une version specifique du model "
        "dont on reprendra l'entrainement",
    )
    parser.add_argument(
        "--preset",
        "--p",
        type=str,
        choices=PRESETS.keys(),
        help="les parametre du LLM predefis pour l'entrainement de chaque modele. "
        "donnés dans le fichier CLI/presets.py",
    )
    parser.add_argument(
        "--max_epochs",
        "--m",
        type=int,
        help="nombre maximum d'epochs sur lequel on veut s'entrainer.",
    )
    parser.add_argument(
        "--time_limit",
        "--time",
        type=int,
        help="temps limite d'entrainement (en minutes) "
        "si une epoche est commencée elle n'est pas interompue par la limite de temps",
    )
    parser.add_argument(
        "--tokenizer_name",
        "--t",
        type=str,
        help="nom du tokenizer que l'on souhaite utiliser (dans le dossier tokenizer_save) "
        "si il n'existe pas, cela vas le creer automatiquement",
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
    parser.add_argument(
        "--wandbOff",
        action="store_true",
        help="permet de desactiver wandb pour l'entrainement",
    )

    args = parser.parse_args()
    assert not (
        args.absolute_gcode and args.relative_gcode
    ), f"you can't use absolut and relative at the same time"

    import torch, gc
    from dataset import svg_dataset
    import tokenizer_pfe.tokenizer_project as tokenizerLib
    from paths_cfg import TOKENIZER_SAVE_DIRECTORY
    from LLM.model import Model, Verbose

    tStart = datetime.now()
    train_cli(
        dataset_path=args.dataset_path,
        save_name=args.save_name,
        preset=args.preset,
        max_epochs=args.max_epochs,
        time_limit=args.time_limit * 60,
        tokenizer_name=args.tokenizer_name,
        absolute_gcode=args.absolute_gcode,
        relative_gcode=args.relative_gcode,
        versionID=args.versionID,
        wandb=(not args.wandbOff),
    )
    print(colored(f"Total time: {prettyTime(datetime.now() - tStart)}", "blue"))
