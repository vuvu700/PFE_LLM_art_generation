# PFE_LLM_art_generation

## instalation

pour installer les dependences il est fortement conseillé d'utiliser UV: (choisir entre extra=gpu ou extra=cpu)
```console
$ uv sync --extra=gpu 
```

Pour executer en ligne de commande le training de model:
'''console
Exemple:
python -m CLI.cli_train --dataset_path dataset/samples_100 --save_name models_1.6M_100 --preset 1.6M --max_epochs 10 --time_limit 20 --tokenizer_name our_tokenizer.json --wandbOff
'''
Cette commande vas lancer un entrainement sur le dataset "samples_100", avec un model de 1.6M parametres, pendent au max 10 epoques ou 20 minutes (une epoch commencée n'est pas interompue)
Il vas sauvgarder les checkpoints du model dans un dossier nommé "models_1.6M" (qui vas contenir tout l'historique des checkpoints, a posteriori ils peuvent etre suprimés sans consequences)
Si le tokenizer ciblé n'existe pas deja, il sera entrainé puis sauvgardé automatiquement.
Les noms de sauvgardes (save_name et tokenizer_name) peuvent etre parametrés comme on veut.
les chemains extactes ou sont sauvgardé les differentas choses seront indiqués a chaque fois dans la console.

pour ajouter des presets ils sont modifiable simplement dans le fichier "./CLI/presets.py"
Ajout ou modifications necessaire pour modifier les parametres du LLM.


Pour executer en ligne de commande le generate des svg:
'''console
Exemple:
python -m CLI.cli_generate --dataset_path dataset/samples_100 --save_generate new_svg  --model_name model_5.5M_1K --version_ID 8 --N 25 --time_limit 3 --top_k 1 --max_tokens 10000000
'''

Pour voir la description des parametres de chacun des fichier CLI:
Python -m CLI.cli_train -h
python -m CLI.cli_generate -h