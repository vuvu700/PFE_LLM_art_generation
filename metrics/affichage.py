import wandb

from metrics.historique import Historique


def affiche_metrics(historique: Historique, name_metrics: None | str | list[str]= None, run_name: None | str = None, id: None | str = None):
    """
    Affiche les metrics sur wandb. Choix de la manière d'on on veut afficher les courbes directement sur le site.
    On peut telecharger ces courbes. 
    Affiche aussi un tableau de la taille du nombre d'epochs. On peut choisir d'afficher les valeurs des metrics et les commentaires.
    Le tout peut etre telecharger aussi sous format csv.

    Entree:
        - historique: la classe Historique
        - name_metrics: str ou list[str]
        - run_name: str, nom que l'on donne au fichier
        - id: str

    Attention: L'id sert pour de la manipulation de run, elle peut etre generer automatiquement.
    Si lancer alors que l'id existe deja, cela ne marchera pas.
    """
    wandb.init(project='pfe', name=run_name, id= id)

    metrics = historique.get_all_historique()
    comments = historique.get_all_commentaries()
    epochs = set()

    for metric in metrics.values():
        epochs.update(metric.keys())
    
    epochs = sorted(epochs)

    if name_metrics is None:
        names = []
    elif isinstance(name_metrics, str):
        names = [name_metrics]
    else:
        names = name_metrics

    columns = ["epoch_id", "comment"] + names
    commentary_table = wandb.Table(columns=columns)

    for epoch in epochs:
        log_data = {}

        for metrics_name, values in metrics.items():
            if epoch in values:
                log_data[metrics_name] = values[epoch]

        epoch_comments = comments.get(epoch, [""])

        for commentary in epoch_comments:
            row = [epoch, commentary]

            if isinstance(name_metrics, str):
                row.append(metrics.get(name_metrics, {}).get(epoch))

            elif isinstance(name_metrics, list):
                for name in name_metrics:
                    row.append(metrics.get(name, {}).get(epoch))

            commentary_table.add_data(*row)

        wandb.log(log_data, step=epoch)

    wandb.log({"all_commentaries": commentary_table})
    wandb.finish()


def affiche_commentaries(historique: Historique, run_name: None | str = None,id: None | str = None):
    """
    Cette fonction permet aussi l'affichage des metrics. Mais aussi l'affichage un tableau de la taille du nombre de commentaires.
    Pour voir les commentaires plus facilement.
    
    Entree:
        - historique: la classe Historique
        - run_name: str, nom que l'on donne au fichier
        - id: str

    Attention: L'id sert pour de la manipulation de run, elle peut etre generer automatiquement.
    Si lancer alors que l'id existe deja, cela ne marchera pas.
    """
    wandb.init(project='pfe', name=run_name, id=id)

    metrics = historique.get_all_historique()
    comments = historique.get_all_commentaries()
    epochs = set()

    for m in metrics.values():
        epochs.update(m.keys())

    epochs = sorted(epochs)

    comment_table = wandb.Table(columns=["epoch", "comment"])

    for epoch in epochs:
        log_data = {}

        for metric_name, values in metrics.items():
            if epoch in values:
                log_data[metric_name] = values[epoch]

        if epoch in comments:
            for comment in comments[epoch]:
                comment_table.add_data(epoch, comment)

        wandb.log(log_data, step=epoch)

    wandb.log({"comments_table": comment_table})
    wandb.finish()