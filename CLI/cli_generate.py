
from pathlib import Path
import torch, gc
from termcolor import colored
import colorama
colorama.init()

from argparse import ArgumentParser
from datetime import datetime

from holo.prettyFormats import prettyTime
from holo.pointers import Pointer

from dataset import svg_dataset
from LLM.model import Model, GenerationStats

def generate_cli(dataset_path: Path, save_generate: str, model_name: str, version_ID: int, N_start: int | None, time_limit: int | None, top_k: int | None, max_tokens: int | None):
    '''
    Boucle pour la generation de svg en ligne de commande.
    Exemple d'execution(a la racine): 
    python -m CLI.cli_generate --dataset_path dataset/samples_100 --save_generate new_svg  --model_name model_5.5M_1K --version_ID 8 --N 25 --time_limit 3 --top_k 1 --max_tokens 10000000

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
    model = Model.load(model_name, versionID=version_ID, device=torch.device("cuda:0"))
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

    statsPtr: Pointer[GenerationStats] = Pointer()
    save_generate_path = Path("repository_svg") / save_generate
    with open(save_generate_path, "w") as f:
        if start is not None:
            f.write(start)

        for txt in model.generate_flow(
                start=start, decode_batch=256, temperature=1.0, top_k=top_k, 
                max_tokens=max_tokens, max_time=time_limit, statsPtr=statsPtr):
            f.write(txt)

    print("start: \n",start)
    print(colored(statsPtr.value, "blue"))
    print(colored(f" -> {statsPtr.value.nb_tokens / statsPtr.value.gen_time:.2f} tokens/sec", "blue"))
 

if __name__ == "__main__":
    parser = ArgumentParser(description="Generating Model")
    parser.add_argument('--dataset_path', type=Path,  help="Chemin du dataset")
    parser.add_argument('--save_generate', type=str,  help="Non du fichier svg que l'on veut sauvegarder (dans le dossier repository_svg)")
    parser.add_argument('--model_name', type=str, help="nom du model a load")
    parser.add_argument('--version_ID', type=int, help="numero de version du model")
    parser.add_argument('--N', type=int, default=None, help="le fichier sur lequel on veut que l'IA continue d'ecrire, (non specifier genere un fichier a partir de rien)")
    parser.add_argument('--time_limit', type=int, default=None, help="Limite de temps en minutes (pas specifier pas de limite de temps)")
    parser.add_argument('--top_k', type=int, default=None, help="les k meilleurs pour la generation du model (pas specifier, n'utilise pas de top k)")
    parser.add_argument('--max_tokens', type=int, default=None, help="la limite de tokens a generer (pas specifier aucune limite)")

    args = parser.parse_args()

    tStart = datetime.now()
    generate_cli(
        dataset_path=args.dataset_path,
        save_generate=args.save_generate,
        model_name=args.model_name,
        version_ID=args.version_ID,
        N_start=args.N,
        time_limit=args.time_limit * 60,
        top_k=args.top_k,
        max_tokens=args.max_tokens
    )
    print(colored(f"Total time: {prettyTime(datetime.now() - tStart)}", "blue"))