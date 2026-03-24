import os

os.environ["WANDB_SILENT"] = "true"

import wandb
import wandb.errors

from metrics.historique import Historique

WANDB_LOGGED = False


def wandb_login():
    global WANDB_LOGGED
    if WANDB_LOGGED is False:
        wandb.login()


def affiche_metrics(historique: Historique, run_name: str, run_ID: str) -> None:
    """
    affiche toutes les metrics de l'historique sur la run choisit\n

    args:
    historique: les metrics a afficher
    run_name: le nom de la run
    run_ID: l'ID qui la rend unique
      si une run avec le meme ID existe deja -> update cette run
      si il n'y a pas de run avec ce nom -> en crée une nouvelle
    """
    wandb_login()
    api = wandb.Api()
    run_existed = True
    try:
        run = api.run(f"pfe_projet-organization/pfe/{run_ID}")
        run_existed = True
    except wandb.errors.errors.CommError:
        run_existed = False

    if run_existed:
        update_affiche_metrics(historique=historique, run_ID=run_ID)
    else:
        init_affiche_metrics(historique=historique, run_name=run_name, run_ID=run_ID)


def init_affiche_metrics(historique: Historique, run_name: str, run_ID: str):
    """fait une nouvelle run et affiche toutes les nouvelles metrics de l'historique de cette run\n

    args:
    `historique`: les metrics a afficher
    `run_name`: le nom de la run
    `run_ID`: l'ID qui la rend unique (doit deja correspondre a une run existante)
    """
    wandb_login()
    wandb.init(project="pfe", name=run_name, id=run_ID)

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

        wandb.log(log_data, step=int(epoch))

    wandb.log({"comments_table": comment_table})
    wandb.finish()


def update_affiche_metrics(historique: Historique, run_ID: None | str = None):
    """met a jour une run et affiche toutes les nouvelles metrics de l'historique sur la run choisit\n

    args:
    `historique`: les metrics a afficher
    `run_name`: le nom de la run
    `run_ID`: l'ID qui la rend unique (doit deja correspondre a une run existante)
    """
    if run_ID is None:
        raise ValueError("wrong_id")

    wandb.init(project="pfe", id=run_ID, resume="allow")

    metrics = historique.get_all_historique()
    comments = historique.get_all_commentaries()

    comment_table = wandb.Table(columns=["epoch", "comment"])

    try:
        api = wandb.Api()
        run = api.run(f"pfe_projet-organization/pfe/{run_ID}")
        old_table = run.use_artifact("comments_table:latest").get("comments_table")
        for row in old_table.data:
            comment_table.add_data(row["epoch"], row["comment"])

    except Exception:
        pass

    epochs = set()

    for m in metrics.values():
        epochs.update(m.keys())

    epochs = sorted(epochs)

    for epoch in epochs:
        log_data = {}

        for metric_name, values in metrics.items():
            if epoch in values:
                log_data[metric_name] = values[epoch]

        if epoch in comments:
            for comment in comments[epoch]:
                comment_table.add_data(epoch, comment)

        wandb.log(log_data, step=int(epoch))

    wandb.log({"comments_table": comment_table})
    wandb.finish()
