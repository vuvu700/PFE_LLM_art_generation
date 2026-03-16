
from pathlib import Path
from termcolor import colored
import colorama
colorama.init()

from argparse import ArgumentParser
from datetime import datetime

from holo.prettyFormats import prettyTime


PRESETS = {
    "1.6M": dict(depth=6, head_dim=128, context_size=4096, nb_heads_mult=1),
    "5.5M": dict(depth=6, head_dim=256, context_size=4096, nb_heads_mult=1),
    "5.5M_1K": dict(depth=6, head_dim=256, context_size=4096, nb_heads_mult=1),
    "20.5M": dict(depth=6, head_dim=512, context_size=4096, nb_heads_mult=5),
}

def train_cli(dataset_path: Path, save_name: str, preset: str, max_epochs: int, time_limit: 
              int, tokenizer_name: str):
    '''
    Boucle pour generer l'entrainement en ligne de commande.
    Exemple d'execution(a la racine): 
    python -m CLI.cli_train --dataset_path dataset/samples_100 --save_name models_1.6_tests --preset 1.6M --max_epochs 5 --time_limit 15 --tokenizer_name our_tokenizer.json

    Pour voir la description des parametres:
    python -m CLI.cli_train -h

    dataset_path: le chemin du dataset
    save_name: nom de sauvegarde du modele
    preset: fixations de parametre du LLM predefis pour l'entrainement de chaque modele. ATTENTION: le definir avant
    max_epochs: nombre maximum d'epochs sur lequel on veut s'entrainer.
    time_limit: temps limite d'entrainement (en minutes)
    tokenizer_name: nom du tokenizer que l'on souhaite utiliser (dans le dossier tokenizer_save)
    '''
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    print(colored("dataset_path valid", "green"))

    tokenizer_path = Path("tokenizer_save") / tokenizer_name

    if not tokenizer_path.exists():
        print(colored("tokenizer not found", "red"))
        answer = input("train tokenizer? (y/n): ")

        if answer.lower() == "y":
            dataset = svg_dataset.SVGDataset(dataset_path, context_size=4096)

            tokenizer = tokenizerLib.Tokenizer.train_from_iterator(
                (svg.txt for svg in dataset.samples),vocab_size=1024,
                special_tokens=tokenizerLib.SPECIAL_TOKENS)
            
            tokenizer.save(tokenizer_path)
            del dataset
            gc.collect()
            torch.cuda.empty_cache()

        else:
            print(colored("fail to import tokenizer", "red"))
            return

    else:
        print(colored("loading tokenizer", "green"))
        tokenizer = tokenizerLib.Tokenizer.load(tokenizer_path) # type: ignore

    preset_config = PRESETS[preset]

    dataset = svg_dataset.SVGDataset(
        dataset_path,
        context_size=preset_config["context_size"],
        tokenizer=tokenizer.encode,
        decoder=tokenizer.decode
    )

    model = Model(save_name=save_name,depth=preset_config["depth"], head_dim= preset_config["head_dim"], 
        context_size=preset_config["context_size"], nb_heads_mult= preset_config["nb_heads_mult"] ,
        tokenizer=tokenizer_path, device="cuda")
    model.show_infos()

    print(colored("start training", "green"))

    model.train(dataset=dataset, batch_size=8, nbEpoches=max_epochs, timeLimite=time_limit, verbose=Verbose.liveProgress)
    
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
    parser.add_argument('--dataset_path', type=Path,  help="Chemin du dataset")
    parser.add_argument('--save_name', type=str, help="nom du model a sauvegarder")
    parser.add_argument('--preset', type=str, choices=PRESETS.keys(), help="choix du model pour le preset de config")
    parser.add_argument('--max_epochs', type=int,  help="maximum d'epochs d'entrainement")
    parser.add_argument('--time_limit', type=int, help="Limite de temps en minutes, (finit l'epoch sur lequel le model s'entraine avant de s'arreter)")
    parser.add_argument('--tokenizer_name', type=str,  help="Nom du tokenizer a utiliser (dans le dossier tokenizer_save)")

    args = parser.parse_args()
    
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
        tokenizer_name=args.tokenizer_name
    )
    print(colored(f"Total time: {prettyTime(datetime.now() - tStart)}", "blue"))