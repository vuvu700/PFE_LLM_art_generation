# PFE_LLM_art_generation

## instalation

les chemain de sauvgarde pour les models/tokenizers/geneations peuvent etre changés grace a un fichier ".env" comme ci dessous
```ini
TOKENIZER_SAVE_DIRECTORY="... chemain vers le dossier"
MODELS_SAVE_DIRECTORY="..."
GENERATIONS_DIRECTORY="..."
```

pour installer les dependences il est fortement conseillé d'utiliser UV: (choisir entre extra=gpu ou extra=cpu)
```bash
$ uv sync --extra=gpu 
```

Pour executer en ligne de commande le training de model:
```bash
Exemple:
$ python -m CLI.cli_train --dataset_path dataset/samples_100 --save_name model_1.6M_100 --preset 1.6M --max_epochs 10 --time_limit 20 --tokenizer_name our_tokenizer.json --wandbOff
```
Cette commande vas lancer un entrainement sur le dataset "samples_100", avec un model de 1.6M parametres, pendent au max 10 epoques ou 20 minutes (une epoch commencée n'est pas interompue).
Il vas sauvgarder les checkpoints du model dans un dossier nommé "model_1.6M" (qui vas contenir tout l'historique des checkpoints, a posteriori ils peuvent etre suprimés sans consequences).
Si le tokenizer ciblé n'existe pas deja, il sera entrainé puis sauvgardé automatiquement.
Les noms de sauvgardes (save_name et tokenizer_name) peuvent etre parametrés comme on veut.
les chemains extactes ou sont sauvgardé les differentas choses seront indiqués a chaque fois dans la console.

pour ajouter des presets ils sont modifiable simplement dans le fichier "./CLI/presets.py".


Pour executer en ligne de commande le generate des svg:
```bash
Exemple:
$ python -m CLI.cli_generate --start_file dataset/samples_100/0045_circle_packing.svg --save_generate svg_generated  --model_name model_1.6M --version_ID 8 --time_limit 240 --temperature 1.0  --top_k 10 --max_tokens 50000
```
cette comande vas lancer la generation d'un fichier SVG, pour continuer le fichier n°45 du dataset_100 en utilisant le model 1.6M et sa version n°8. Le temps limite de generation est 240 secondes ou 50000 tokens génerés. Il vas generer en utilisant une temperature de 0.8 et un topK de 7.
Si --start_file n'est pas donné, il vas commencer un nouveau svg.
le fichier generé est nommé "svg_generated" comme demandé et sera sauvgardé dans le dossier "repository_svg" si le .env ne l'as pas respecifié.

Pour voir la description des parametres de chacun des fichier CLI:
```bash
$ python -m CLI.cli_train -h
$ python -m CLI.cli_generate -h
```