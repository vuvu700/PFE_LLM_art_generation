import time
import random
from collections import defaultdict
from holo.profilers import Profiler

import torch
import torch.nn.functional as F

from tokenizer_pfe.tokenizer_project import Tokenizer
from dataset.svg_dataset import _Tokens, SVGDataset, IGNORE_INDEX

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


class ChunckAssembler:
    def __init__(
        self,
        tokenizer: Tokenizer,
        context_size: int,
        temperature: float,
        top_k: int | None,
    ) -> None:
        """
        args:
            `tokenizer`: le tokenizer utilisé pour decoder les tokens samplés
            `context_size`: taille nominale du context
            `temperature` et `top_k`: utilisés pour le sampling des logits
        """
        assert context_size % 2 == 0, f"la context_size({context_size}) doit etre paire"
        assert (top_k is None) or (
            (top_k > 0) and (top_k < tokenizer.get_vocab_size())
        ), (
            f"la valeur de top_k({top_k}) n'est pas addaptée:"
            f" doit etre 1 <= ... < {tokenizer.get_vocab_size()}"
        )
        assert temperature >= 0.0, f"temerature invalide: {temperature!r}"
        self.tokenizer: Tokenizer = tokenizer
        self.context_size: int = context_size
        self.temperature: float = temperature
        self.top_k: int | None = top_k
        self._stored_chuncks: dict[int, dict[int, str]] = defaultdict(dict)
        """{svgIndex -> {chunckIndex -> text to be assembled}}\n
        le text deja decoupé pour l'assemblage 
        (peut etre vide pour le dernier chunck si len() < `context_size`//2)"""
        self._prof = Profiler(["sample", "decode", "assemble"])

    def __sample(self, logits: torch.Tensor) -> list[int]:
        """sampling des logits (pas par batch)\n
        args:
            `logits`: les logits a sampler (nbTokens, vocab_size)
        return: la liste des tokens
        """
        # unitées de temps données a partir de tests avec context_size=4096
        if (self.top_k is not None) and (self.top_k > 0):
            # 100us
            topk_logits, _ = torch.topk(logits, self.top_k, dim=-1)  # (ctx, topK)
            logits.masked_fill_(logits < topk_logits[:, -1:], float("-inf"))
        if self.temperature > 0:
            # 200us
            probs = F.softmax(logits / self.temperature, dim=-1)
            tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:  # 30us
            tokens = torch.argmax(logits, dim=-1)
        return tokens.tolist()  # 400us
        # TODO: partie la plus lente, a optimizer (! rester sur GPU !)
        #  -> decode est ++ rapide avec list[int] que numpy
        #   et .cpu().numpy() est plus lent car pas list[int],
        #   ajouter .tolist() apres ne l'as pas rendu significativement
        #   + rapide pour les petit ctx, mais pour ctx~100K 20% de gains

    @torch.no_grad
    def add_logits(
        self,
        batch_logits: torch.Tensor,
        svgIndexes: list[int],
        chunckIndexes: list[int],
    ) -> None:
        """decoupe, sample et decode les logits donnés,
        puis les stock en attendant l'assemblage\n
        args:
            `batch_logits`: le batch de logits qui viens d'etre generé
            `svgIndexes`: les index des svg du batch
            `chunckIndexes`: les index des chuncks du batch
        """
        # TODO: prendre en compte les IGNORE_INDEX
        #   -> ajouter un param targets (batch, context_size)[int64]
        #   cut les `logits` qui ont une target == IGNORE_INDEX
        # process each chunck of the batch
        batch_size, ctx_size, vocab_size = batch_logits.shape
        for iBatch in range(batch_size):
            svgIdx: int = svgIndexes[iBatch]
            chIdx: int = chunckIndexes[iBatch]
            logits: torch.Tensor = batch_logits[iBatch]  # (ctx_size, vocab_size)
            if chIdx != 0:
                # => not first chunck of svg, only keep the last half
                logits = logits[self.context_size // 2 :]
            if logits.size(0) == 0:
                # => empty beacause the chunck is too small
                self._stored_chuncks[svgIdx][chIdx] = ""
                continue  # finished with this chunck
            # => non empty logits, sample from them
            with self._prof.mesure("sample"):
                tokens = self.__sample(logits)
            # decode the tokens
            with self._prof.mesure("decode"):
                decoded = self.tokenizer.decode(tokens)
                assert isinstance(decoded, str), f"unexpected type: {type(decoded)}"
                self._stored_chuncks[svgIdx][chIdx] = decoded

    def assemble_chuncks(self) -> dict[int, str]:
        """assemble les chuncks decodés qui sont stockés\n
        return: {svgIndex -> text assemblé du svg en question}
        """
        assembled: dict[int, str] = {}
        # ensure there are no chunck missing
        for svgIndex, chuncks in self._stored_chuncks.items():
            chunckIndexes: list[int] = sorted(chuncks.keys())
            assert chunckIndexes == list(range(len(chunckIndexes))), (
                f"unexpected chuncks index (expected: {range(len(chunckIndexes))}) "
                f"but got {chunckIndexes}"
            )
            # => got all the chuncks of the svg, can assemble them
            with self._prof.mesure("assemble"):
                assembled[svgIndex] = "".join(
                    (chuncks[chIdx] for chIdx in chunckIndexes)
                )
        return assembled
