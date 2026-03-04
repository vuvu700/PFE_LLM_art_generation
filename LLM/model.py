from .nanochat.gpt import GPT, DistMuonAdamW, MuonAdamW
from tokenizer_pfe.tokenizer_project import Tokenizer
from metrics.historique import Historique


class Model():
    __slots__ = ("llm", "tokenizer", "optimizer", "historique", )
    
    llm: GPT
    tokenizer: Tokenizer
    optimizer: DistMuonAdamW|MuonAdamW
    historique: Historique
    
    def __init__(self) -> None:
        pass # TODO (project issue #38)