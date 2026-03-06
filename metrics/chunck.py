import torch
import torch.nn.functional as F
from tokenizer_pfe.tokenizer_project import Tokenizer
from dataset import svg_dataset
import random
import time

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
        dataset: svg_dataset.SVGDataset,
        tokenizer: Tokenizer,
        context_size: int,
        vocab_size: int,
        temperature=1.0,
        top_k=None,
        seed=-1,
        batch_size=5000
    ) -> tuple[dict[int, str], dict[int, list[int]]]:
    """
    Assemble les chunks de tokens de chaque SVG puis les decodes.
    On traite les SVG par batch. Si batch trop petite, risque de crash pour cause de probleme de memoire.
    On decode les SVG un par un.
    
    Entrée:
        dataset: Dataset contenant les chunks
        tokenizer: Tokenizer
        contextesize: int doit etre pair
        temperature: int
        top_k: int
        seed: int
        batch_size: int

    Sortie:
        svgs: {svgID -> text decodé}
        svgs_tokens: {svgID -> tokens samplés}
    """

    assert context_size % 2 == 0, "context_size doit être pair"

    svgs_text = {}
    svgs_tokens = {}
    svg_chunks = {}

    for chunk in dataset.chunks:
        svgIndex = chunk.indexes.svgIndex
        if svgIndex not in svg_chunks:
            svg_chunks[svgIndex] = []
        svg_chunks[svgIndex].append(chunk)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for svgIndex, chunks in svg_chunks.items():
        tokens = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                tokens.extend(chunk.tokens)
            else:
                tokens.extend(chunk.tokens[context_size // 2:])

        svg_tokens = []

        for index in range(0, len(tokens), batch_size):
            batch_tokens = tokens[index:index + batch_size]
            seq_len = len(batch_tokens)

            chunk_logits = torch.full((1, seq_len, vocab_size), -torch.inf, device=device)
            idx_tensor = torch.tensor(batch_tokens, device=device)
            chunk_logits[0, torch.arange(seq_len, device=device), idx_tensor] = 0.0

            sampled_tokens = sampling_logits(chunk_logits, temperature, top_k, seed)
            svg_tokens.extend(sampled_tokens)

            
            del chunk_logits, sampled_tokens
            torch.cuda.empty_cache()

        svgs_tokens[svgIndex] = svg_tokens
        svgs_text[svgIndex] = tokenizer.decode(svg_tokens)
        print(svgIndex ,'is decoded')
        
    return svgs_text, svgs_tokens