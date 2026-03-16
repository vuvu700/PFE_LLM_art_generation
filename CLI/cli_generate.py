
from pathlib import Path
from termcolor import colored
import colorama
colorama.init()

from argparse import ArgumentParser
from datetime import datetime

from holo.prettyFormats import prettyTime, SingleLinePrinter
from holo.pointers import Pointer

from paths_cfg import GENERATIONS_DIRECTORY


def generate_cli(dataset_path: Path, save_generate: str, model_name: str, version_ID: int, N_start: int | None, time_limit: int | None, top_k: int | None, max_tokens: int | None):
    '''
    Boucle pour la generation de svg en ligne de commande.
    Exemple d'execution(a la racine): 
    python -m CLI.cli_generate --dataset_path dataset/samples_100 --save_generate new_svg  --model_name model_5.5M_1K --version_ID 8 --N 25 --time_limit 3 --top_k 1 --max_tokens 10000000

    Pour voir la description des parametres:
    python -m CLI.cli_generate -h
    

    dataset_path: le chemin du dataset
    save_generate: nom du fichier svg que l'on souhaite generer
    model_name: nom du modele que l'on veut load
    version_ID: version de l'id du model ( par rapport au nombre d'epochs entrainer)
    N_start: choix du fichier dans le dataset sur lequel l'IA va commencer a ecrire
    time_limit: temps limite d'entrainement (en minutes)
    top_k: choix du top_k pour le model
    max_tokens: max des tokens que l'on ne souhaite pas depasser
    '''
    try:
        torch.cuda.empty_cache()
        del model # type: ignore
        torch.cuda.empty_cache()
    except Exception: pass
    model = Model.load(model_name, versionID=version_ID, device=torch.device("cuda:0"), compile=False)
    model.show_infos()

    print(f"trained for {model.nb_epoches_done} epoches")
    for k, v in model.historique.get_all_historique().items():
        print(colored(f"{k}: {v.get((model.nb_epoches_done-1), None):.4g}", "green")) # type: ignore


    if (N_start != None) :
        print(colored("loading dataset", "blue"))
        dataset = svg_dataset.SVGDataset(
            dataset_path, context_size=model.context_size,
            tokenizer=model.tokenizer.encode, decoder=model.tokenizer.decode)
        print(colored(f"using: {dataset.samples[N_start].svg_file}", "blue"))
        start = dataset.samples[N_start].txt[: model.context_size]
        del dataset
        gc.collect()

    else:
        start = None

    print(colored("start generating", "blue"))
    statsPtr: Pointer[GenerationStats] = Pointer()
    save_generate_path = GENERATIONS_DIRECTORY / save_generate
    with open(save_generate_path, "w") as f:
        singleLine = SingleLinePrinter(None)
        if start is not None:
            f.write(start)

        for txt in model.generate_flow(
                start=start, decode_batch=64, temperature=1.0, top_k=top_k,
                max_tokens=max_tokens, max_time=time_limit, statsPtr=statsPtr):
            f.write(txt)
            
            singleLine.print(
                f"progress: {statsPtr.value.nb_tokens:_d} tokens generated "
                f"(running for {prettyTime(statsPtr.value.gen_time)})")
        singleLine.newline()
    
    print(colored(f"finished generating ({statsPtr.value.stop_reason})", "green"))
    print(colored(f" -> {statsPtr.value.nb_tokens / statsPtr.value.gen_time:.2f} tokens/sec", "blue"))
 

if __name__ == "__main__":
    parser = ArgumentParser(description="Generating Model")
    parser.add_argument('--dataset_path', "--d", type=Path, required=True, help="Chemin du dataset")
    parser.add_argument('--save_generate', "--s", type=str, required=True, help="Non du fichier svg que l'on veut sauvegarder (dans le dossier repository_svg)")
    parser.add_argument('--model_name', "--m", type=str, required=True, help="nom du model a load")
    parser.add_argument('--version_ID', "--id", type=int, required=True, help="numero de version du model")
    parser.add_argument('--N', type=int, default=None, help="le fichier sur lequel on veut que l'IA continue d'ecrire, (non specifier -> genere un fichier a partir de rien)")
    parser.add_argument('--time_limit', "--t", type=int, default=None, help="Limite de temps en secondes (pas specifier -> pas de limite de temps)")
    parser.add_argument('--top_k', "--k", type=int, default=None, help="les k meilleurs pour la generation du model (pas specifier -> n'utilise pas de top k)")
    parser.add_argument('--max_tokens', "--l", type=int, default=None, help="la limite de tokens a generer (pas specifier -> aucune limite)")

    args = parser.parse_args()
    
    import torch, gc
    from dataset import svg_dataset
    from LLM.model import Model, GenerationStats

    tStart = datetime.now()
    generate_cli(
        dataset_path=args.dataset_path,
        save_generate=args.save_generate,
        model_name=args.model_name,
        version_ID=args.version_ID,
        N_start=args.N,
        time_limit=args.time_limit,
        top_k=args.top_k,
        max_tokens=args.max_tokens
    )
    print(colored(f"Total time: {prettyTime(datetime.now() - tStart)}", "blue"))