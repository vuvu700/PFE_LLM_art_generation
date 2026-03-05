import torch
import torch.nn.functional as F
from tokenizer_pfe.tokenizer_project import Tokenizer
from collections import defaultdict
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


def sampling_logits(logits: torch.Tensor, temperature=1.0, top_k=None, seed=-1):
    """
    On passe le logit dans lequel on applique un sampling afin de donner une liste de tokens

    Entrée:
        logits: dict {(svgIndex, chunkIndex): torch.Tensor}
        temperature: int
        top_k: int
        seed: int

    Sortie:
        tokens: Liste de tokens
    """
    if seed >= 0:
        torch.manual_seed(seed)
        random.seed(seed)
    logits = logits.squeeze(0)  # (ctx_size, vocab_size)
    ctx_size, vocab_size = logits.shape
    if (top_k is not None) and (top_k > 0):
        v, _ = torch.topk(logits, min(top_k, vocab_size),
                          dim=-1)  # (ctx, topK)
        logits[logits < v[:, [-1]]] = -float('Inf')
    if temperature > 0:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    else:
        tokens = torch.argmax(logits, dim=-1)
    return tokens.tolist()


def assemble_decode(
        logits: dict[tuple[int, int], torch.Tensor],
        tokenizer: Tokenizer, context_size: int, temperature=1.0,
        top_k=None, seed=-1) -> tuple[dict[int, str], dict[int, list[int]]]:
    """
    Assemble les chunks de tokens de chaque SVG puis les decodes.

    Entrée:
        logits: dict {(svgIndex, chunkIndex): torch.Tensor}
        tokenizer: Tokenizer
        contextesize: int doit etre pair
        temperature: int
        top_k: int
        seed: int

    Sortie:
        svgs: {svgID -> text decodé}
        svgs_tokens: {svgID -> tokens samplés}
    """
    assert context_size%2 == 0, "la contexte size doit etre pair"
    svgs_tokens: dict[int, list[int]] = {}
    svg_text: dict[int, str] = {}
    chunk_dic = {}
    for (svgIndex, chunckIndex), chunk_logits in sorted(logits.items(), key=None):
        tokens = sampling_logits(chunk_logits, temperature, top_k, seed)
        if svgIndex not in chunk_dic:
            chunk_dic[svgIndex] = {}
        
        chunk_dic[svgIndex][chunckIndex] = tokens
    
    for svgIndex in chunk_dic:
        svgs_tokens[svgIndex] = []
        for chunks in sorted(chunk_dic[svgIndex].keys()):
            tokens = chunk_dic[svgIndex][chunks]
            if chunks == 0:
                svgs_tokens[svgIndex].extend(tokens)
            else:
                svgs_tokens[svgIndex].extend(tokens[context_size//2:])

        svg_text[svgIndex] = tokenizer.decode(svgs_tokens[svgIndex])

    return svg_text, svgs_tokens
