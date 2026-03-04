import torch
import torch.nn.functional as F
from tokenizer_pfe.tokenizer_project import Tokenizer
from collections import defaultdict
from typing import Dict, Tuple
import random

"""a partir des `logits` en entrée, fait un sampling dessus et les assemble pour renvoyer les differents svg qu'ils composent\n

    les parametres:
        - `logits`: {(svgIndex, chunckIndex) -> logits of shape (1, CTX_SIZE ou inferieur, VOCAB_SIZE)}
        NOTE: la presence de svgIndex implique que les logits en entrée peuvent provenire de plusieurs fichiers
            toutefois il est garenti que l'ensemble des logits d'un fichier sont donnés en entrée dans `logits` (pas de chunck manquants)
        - `tokenizer`: le tokenizer du projet a utiliser
        - `temperature`, `top_k`, `seed`: temperature, topk, seed pour le sampling des logits

    etapes:
        - input -> LLM -> logits -> sampling -> tokens -> assemblage des chuncks de tokens -> decodage des tokens -> text(svg)
    
    renvois:
        - {svgIndex -> text du svg reconstitué a partir des chuncks assemblés et decodés}

"""

def sampling_logits(logits:torch.Tensor, tokenizer:Tokenizer, temperature=1.0, top_k=None, seed=-1):
    """
    On passe le logit dans lequel on applique un sampling afin de donner une liste de tokens

    Entrée:
        logits: dict {(svgIndex, chunkIndex): torch.Tensor}
        tokenizer: Tokenizer
        temperature: int
        top_k: int
        seed: int

    Sortie:
        tokens: Liste de tokens
    """
    if seed >= 0:
        torch.manual_seed(seed)
        random.seed(seed)
    
    logits = logits.squeeze(0)
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        logits[logits < v[:, [-1]]] = -float('Inf')
    
    if temperature > 0:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    else:
        tokens = torch.argmax(logits, dim=-1)
    return tokens.tolist()


def assemble_decode(logits: Dict[Tuple[int, int], torch.Tensor], tokenizer:Tokenizer, temperature=1.0, top_k=None, seed=-1)-> Dict[int, str]:
    """
    Assemble les chunks de tokens de chaque SVG puis les decodes.
    
    Entrée:
        logits: dict {(svgIndex, chunkIndex): torch.Tensor}
        tokenizer: Tokenizer
        temperature: int
        top_k: int
        seed: int

    Sortie:
        svgs: Dictionnaire de taillle {int, str}
    """
    svgs_tokens = {}
    svgs = {}
    for (svgIndex, chunckIndex), chunk_logits in sorted(logits.items(), key=None):
        tokens = sampling_logits(chunk_logits, tokenizer, temperature, top_k, seed)
        if svgIndex not in svgs_tokens:
            svgs_tokens[svgIndex] = []
        svgs_tokens[svgIndex].extend(tokens)
    
    for svg_idx, tokens in svgs_tokens.items():
        svgs[svg_idx] = tokenizer.decode(tokens)
    
    return svgs