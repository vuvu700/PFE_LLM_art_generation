# PFE_LLM_art_generation

## Installation

Les chemins de sauvegarde pour les models/tokenizers/generations peuvent être changés grâce à un fichier ".env" comme ci dessous.
Si les chemins donnés n'existent pas, ils seront automatiquement créés.
```ini
TOKENIZER_SAVE_DIRECTORY="... chemin vers le dossier"
MODELS_SAVE_DIRECTORY="..."
GENERATIONS_DIRECTORY="..."
```

Pour installer les dépendances, il est fortement conseillé d'utiliser UV: (choisir entre extra=gpu ou extra=cpu)
```bash
$ uv sync --extra=gpu 
```

Pour exécuter en ligne de commande le training du model:
```bash
Exemple:
$ python -m CLI.cli_train --dataset_path dataset/samples_100 --save_name model_1.6M_100 --preset 1.6M --max_epochs 10 --time_limit 20 --tokenizer_name our_tokenizer.json --wandbOff
```
Cette commande va lancer un entrainement sur le dataset "samples_100", avec un model de 1.6M parametres, pendant au max 10 epoques ou 20 minutes (une epoch commencée n'est pas interrompue).
Il va sauvegarder les checkpoints du model dans un dossier nommé "model_1.6M" (qui va contenir tout l'historique des checkpoints, a posteriori ils peuvent être supprimés sans conséquences).
Si le tokenizer ciblé n'existe pas déjà, il sera entrainé puis sauvegardé automatiquement.
Les noms de sauvegardes (save_name et tokenizer_name) peuvent être paramétrés comme on veut.
Les chemins exactes ou sont sauvegardés les différentes choses seront indiqués a chaque fois dans la console.

Pour ajouter des presets, ils sont modifiables simplement dans le fichier "./CLI/presets.py".


Pour exécuter en ligne de commande la génération des svg:
```bash
Exemple:
$ python -m CLI.cli_generate --start_file dataset/samples_100/0045_circle_packing.svg --save_generate svg_generated  --model_name model_1.6M --version_ID 8 --time_limit 240 --temperature 1.0  --top_k 10 --max_tokens 50000
```
Cette commande va lancer la génération d'un fichier SVG, pour continuer le fichier n°45 du dataset_100 en utilisant le model 1.6M et sa version n°8. Le temps limite de génération est 240 secondes ou 50000 tokens générés. Il va générer en utilisant une température de 0.8 et un topK de 7.
Si --start_file n'est pas donné, il va commencer un nouveau svg.
Le fichier généré est nommé "svg_generated" comme demandé et sera sauvegardé dans le dossier "repository_svg" si le .env ne l'a pas spécifié.

Pour voir la description des paramètres de chacun des fichiers CLI:
```bash
$ python -m CLI.cli_train -h
$ python -m CLI.cli_generate -h
```
