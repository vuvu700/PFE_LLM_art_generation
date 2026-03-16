# PFE_LLM_art_generation

## instalation

pour installer les dependences il est fortement conseillé d'utiliser UV: (choisir entre extra=gpu ou extra=cpu)
```console
$ uv sync --extra=gpu 
```

Pour executer en ligne de commande le training de model:
'''console
Exemple:
python -m CLI.cli_train --dataset_path dataset/samples_100 --save_name models_1.6_tests --preset 1.6M --max_epochs 5 --time_limit 15 --tokenizer_name our_tokenizer.json
'''

Commentaires: voir dans le fichier CLI.cli la liste des preset possible.
Ajout ou modifications necessaire pour modifier les parametres du LLM.


Pour executer en ligne de commande le generate des svg:
'''console
Exemple:
python -m CLI.cli_generate --dataset_path dataset/samples_100 --save_generate new_svg  --model_name model_5.5M_1K --version_ID 8 --N 25 --time_limit 3 --top_k 1 --max_tokens 10000000
'''