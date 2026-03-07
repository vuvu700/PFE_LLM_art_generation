"""
methodes to compute the metrics that are used to evaluate the LLM

for the training we will mesure the following metrics:
 - related to the logits:
    - CrossEntropy (CE) [TRAIN] (todo)
        quantifies how uncertain a language model is when predicting the next token in a sequence. 
        (Sensitivity to outliers, depends on the tokenizer)
    - Perplexity (PPL) [TRAIN] (todo)
        like CE, quantifies uncertainty of the next token 
        -> interpreted as the effective number of choices the model has for the next token, averaged over the sequence.
        (Sensitivity to outliers, depends on the tokenizer)
    - Bit-Per-Character (BPC) [TRAIN] (todo)
        quantifies uncertainty of the next character (in text)
        (Sensitivity to outliers, INDEPENDENT of the tokenizer)
        formula = lossCE * NbTokens / (NbCharsInText * log(2))
    NOTE: CE and PPL will also be calculated as CE2 and PPL2 with target = argmax(logits)
    - Prediction Entropy (ENTROPY) [TRAIN + EVAL] (todo)
        formula H = -sum(p_i*log(p_i) for i in range(vocab)), with p the proba distrib
    - logits standard deviation (LOGITS_SD) [TRAIN] (todo)
        the mean SD of the logits (use the dim of the vocab for sd)
    - top-1 & top-k accuracy (TOP-1, TOP-K) [TRAIN] (todo)
        mesure si le token de la target est dans le top K

 - language:
    - Bilingual Evaluation Understudy (BLEU) [TRAIN] (maybe)
        Evaluates n-gram overlap (up to 4). Focuses on precision; penalizes brevity.
    - Recall-Oriented Understudy for Gisting Evaluation (ROUGE) [TRAIN] (maybe)
        Evaluates the specified n-gram overlap. Focuses on recall.

 - others:
    - total nb of tokens (NB_TOKENS) [TRAIN] (todo)
        track the total number of tokens used during that epoch
    - gradient norm (GRAD_NORM) [TRAIN] (todo)
        mesure the gradient of each batch (before croping)
    - Learning Rate (LR) [TRAIN] (done)
        it migth change during training with some scheduling
    - generated size (SIZE_AVG, SIZE_SD) [EVAL] (todo)
        track the average size (and its SD) of the generated files 
        
 - sur la validité du fichier:
    (pour le TRAIN, se base sur les SVG reconstitués)
    (pour le TEST, se baser sur le SVG final generé)
    - Well-formedness rate (XML_VALIDITY) [TRAIN + EVAL] (done)
        % de sorties qui sont du XML valide (parse sans erreur).
    - SVG validity rate (SVG_VALIDITY) [TRAIN + EVAL] (later)
        % de sorties acceptées par un renderer SVG (ex : navigateur, librsvg).
    - Render success rate (RENDER_RATE) [TRAIN + EVAL] (later)
        % de sorties qui produisent effectivement une image non vide.

 - analyses visuels:
    - Structural Similarity Index (SSIM) [TRAIN + EVAL(1?)] (maybe)
        Sensible à la structure visuelle globale.
 
 - sur la structure du fichier.
    - Tag distribution similarity (TAGS_DIST_SIMILARITY) [TRAIN + EVAL(1?)] (later)
        Compare les histogrammes (<path>, <rect>, <circle>…).


NOTE: pour l'evaluation faire 2 methodes
 1) on commence avec un "startOfSequence" seulement
 2) on commence avec le debut d'un des svg du dataset
et pour les deux methodes on le laisse generer un document jusqu'a EndOfSequence ou la limite de tokens
(si on atteins la limite de tokens essayer de finir le SVG -> voir comment s'y prendre)

NOTE: pour les metrics de TRAIN sur les fichiers complets, 
    (/!\\ attention il vas falloir modifier le forward pour renvoyer la loss & les logits)
il faut reconstituer des fichiers a partir des fragments generés 
    (on pouras faire plusieurs samplings a partire des fichiers reconstitués)
 - methode 1) etant donné que les morceaux generés s'entrecroisent
    on vas prendre le premier block complet puis la seconde moitiée de tout les blocks suivants
    de cette façon chaque moitiée béneficie du contexte qui lui precede 
    |-----------|-----|-----|-----|-----|-----|-----|---  (la sequence a reconstituer)
    ############[[[[[[######[[[[[[######[[[[[[######      (les chuncks generés)
          [[[[[[######[[[[[[######[[[[[[######[[[[[[####  (les chuncks generés)
    legende: 
        - "[": premiere moitiée du chunk (non conservés)
        - "#": seconde moitiée du chunck (la partie conservée)
"""
from typing import Literal
from pathlib import Path
from lxml import etree # type: ignore
import math

from holo.profilers import Profiler

import torch
import torch as nn
import torch.nn.functional as F

from LLM.model import Model

############################# metrics accumulator #############################

class MetricsAccumulator():
    """accumulateur pour calculer les metrics suivantes sur une epoche complete.
     - CrossEntropy (CE + CE2)
     - Perplexity (PPL + PPL2)
     - Bit-Per-Character (BPC)
     - Prediction Entropy (ENTROPY)
     - logits standard deviation (LOGITS_SD)
     - top-1 & top-k accuracy (TOP-1 + TOP-K)
     - ... # ajouer les autres metrics
    """
    
    def __init__(self, usage:Literal["train", "val"], topK:int) -> None:
        assert topK > 1
        self.usage = usage
        self.topK: int = topK
        # total metriques
        self.total_CE: float = 0.0
        self.total_CE2: float = 0.0
        self.total_entropy: float = 0.0
        self.total_SD: float = 0.0
        self.total_top1: int = 0
        self.total_topK: int = 0
        # total others
        self.total_tokens: int = 0
        self.total_nbChars: int = 0
        self._prof = Profiler([
            "loss1", "loss2", "CE_related", "entropy", "SD", "topK", "accuacy"])

    # --- logits related metrics ---
    
    def batch_logits_metrics(
            self, logits:torch.Tensor,
            targets:torch.Tensor, totalNbChars:int)->torch.Tensor:
        """args: (all tensors will be detached)
        - logits: les logits du model (batch, ctx, vocab)[Float]
        - targets: les token cibles a predire (batch, ctx)[Long]
        - totalNbChars: pour chaque context du batch, le nb de char dans le text
        return: the Cross-Entropy loss of the model"""
        logits_flatten = logits.view(-1, logits.size(-1))
        targets_flatten = targets.view(-1)
        logits_detached = logits.detach()
        # compute the losses
        with self._prof.mesure("loss1"):
            CE_loss: torch.Tensor = torch.nn.functional.cross_entropy(
                logits_flatten, targets_flatten, ignore_index=-1, reduction="mean")
            # CE_loss.item()
        with self._prof.mesure("loss2"):
            CE2_loss: torch.Tensor = torch.nn.functional.cross_entropy(
                logits_flatten, logits_flatten.argmax(-1), ignore_index=-1, reduction="mean")
            CE2_loss = CE2_loss.detach()
        # CE related
        with self._prof.mesure("CE_related"):
            batch_size, ctx_size, vocab_size = logits_detached.shape
            totalNbTokens: int = (ctx_size * batch_size)
            self.total_tokens += totalNbTokens
            self.total_nbChars += totalNbChars
            self.total_CE += float(CE_loss.detach()) * totalNbTokens
            self.total_CE2 += float(CE2_loss) * totalNbTokens
        # entropy
        with self._prof.mesure("entropy"):
            log_probs = F.log_softmax(logits_detached, dim=-1)
            probs = torch.exp(log_probs)
            self.total_entropy += -float((probs * log_probs).sum())
        # standard deviation
        with self._prof.mesure("SD"):
            self.total_SD += float(logits_detached.std(dim=-1).sum())
        # top-k accuracy
        with self._prof.mesure("topK"):
            sorted_indices = torch.topk(logits_detached, self.topK, dim=-1)[1]
            expanded_targets = targets.unsqueeze(-1).expand(-1, -1, self.topK)
        with self._prof.mesure("accuacy"):
            corrects = torch.eq(sorted_indices, expanded_targets)
            self.total_top1 = int(torch.sum(corrects[:, :, 0]))
            self.total_topK = int(torch.sum(corrects))
        return CE_loss
    
    # --- to get the result ---
    
    def get_metrics(self)->dict[str, float]:
        """return the current state of the different metrics accumulated this epoch"""
        CE = (self.total_CE / self.total_tokens)
        CE2 = (self.total_CE2 / self.total_tokens)
        tokensPerChar = (self.total_tokens / self.total_nbChars)
        metrics = {
            f"CE": CE, f"PPL": math.exp(CE),
            f"BPC": (CE / math.log(2.0)) * tokensPerChar,
            f"CE2": CE2, f"PPL2": math.exp(CE2),
            f"ENTROPY": (self.total_entropy / self.total_tokens),
            f"LOGITS_SD": (self.total_SD / self.total_tokens),
            f"TOP-1": (self.total_top1 / self.total_tokens),
            f"TOP-{self.topK}": (self.total_topK / self.total_tokens)}
        return {f"{name}_{self.usage}": value for name, value in metrics.items()}



############################# other metrics functions #############################


def get_learning_rates(model:Model):
    names = [
        "lm_head", "embedding",  "value_embeds", "residuals", "x0", 
        ] + [f"transformers_grp_{i}" for i in range(4)]
    return {f"lr_{names[i]}": float(optim['lr'])
            for i, optim in enumerate(model.optimizer.param_groups)}

    
############################# file validity metrics functions #############################


def svg_is_fatal(my_svg: str):
    """
    Permet de voir si on peut ouvrir un fichier ou pas.
    Si True, il y a donc un probleme sur le fichier.
    Input: str
    Output: Bool
    """
    try:
        parser = etree.XMLParser()
        tree = etree.parse(my_svg, parser)
        root = tree.getroot()
    except etree.XMLSyntaxError:
        return True
    return False

def svg_nb_errors(my_svg: str):
    """
    Permet de compter le nombre d'erreurs fatales et non fatales d'un fichier svg.
    Une erreur est considerer comme fatal si on n'arrive pas a ouvrir le fichier svg a cause de cette erreur.
    L'implementation des erreurs non fatal etant absentes du fait de la non pertinance des resultats, la valeur renvoyer est None.
    Input: str
    Output: (int, None)
    """
    nb_fatal_errors = 0
    parser = etree.XMLParser()
    try:
        tree = etree.parse(my_svg, parser)
        root = tree.getroot()
    except etree.XMLSyntaxError:
        nb_fatal_errors = len(parser.error_log)
        return nb_fatal_errors
    return 0


