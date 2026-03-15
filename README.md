# PFE_LLM_art_generation

## instalation

pour installer les dependences il est fortement conseillé d'utiliser UV: (choisir entre extra=gpu ou extra=cpu)
```console
$ uv sync --extra=gpu 
```
pour executer en ligne de commande:
'''console
$ python -m CLI.cli dataset_path save_name preset max_epochs time_limit tokenizer_name
Exemple:
python -m CLI.cli dataset/samples_100 models_1.6_tests 1.6M 5 15 our_tokenizer.json
'''
Commentaires: voir dans le fichier CLI.cli la liste des preset possible.
Ajout ou modifications necessaire pour modifier les parametres du LLM.