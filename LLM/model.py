from typing import Literal
from pathlib import Path

import torch

from .nanochat.gpt import GPT, GPTConfig, DistMuonAdamW, MuonAdamW

from tokenizer_pfe.tokenizer_project import Tokenizer
from metrics.historique import Historique


_Device = Literal["cpu", "cuda"]
_WPatternChr = Literal["S", "L"]
_WPattern = tuple[_WPatternChr, ...]

class Model():
    __slots__ = ("config", "llm", "tokenizer", "optimizer", "historique", )
    
    config: GPTConfig
    llm: GPT
    tokenizer: Tokenizer
    optimizer: DistMuonAdamW|MuonAdamW
    historique: Historique
    
    def __init__(
            self, tokenizer:Tokenizer|Path, 
            device:_Device, depth:int, head_dim:int,
            context_size:int, nb_heads_mult:float=1.0) -> None:
        ### --- tokenizer ---
        if isinstance(tokenizer, Path):
            tokenizer = Tokenizer.load(tokenizer)
        self.tokenizer = tokenizer
        ### --- historique ---
        self.historique = Historique()
        ### --- LLM, config, optimizer---
        llm, config = self.__build_model_meta(
            depth=depth, head_dim=head_dim, nb_heads_mult=nb_heads_mult,
            vocab_size=self.vocab_size, effective_sequence_size=context_size,
            window_pattern=("S", "S", "S", "L"))
        self.llm = llm; self.config = config; del llm, config
        # -> All tensors get storage on target device but with uninitialized
        self.llm.to_empty(device=device)
        # -> All tensors get initialized
        self.llm.init_weights()
        # -> get the optimizer
        self.optimizer = self.llm.setup_optimizer()
        # -> compress the LLM to bf16
        self.llm = self.llm.bfloat16()
        # -> compile the LLM
        #   the inputs to model will never change shape so dynamic=False is safe
        self.llm = torch.compile(self.llm, dynamic=False) # type: ignore
        
    def show_infos(self)->None:
        print(self.config)
        params = self.llm.num_scaling_params()
        params_Embed = (params['wte'] + params['value_embeds'])
        print(f"{params['total']:_d} total params "
            f"(embeding: {params_Embed:_d} | "
            f"last layer: {params['lm_head']:_d} | "
            f"transformer: {params['transformer_matrices']:_d})")
        print(f"on device: {self.device!r}, with effective context_size: {self.context_size}")
    
    @staticmethod
    def __build_model_meta(
            depth:int, head_dim:int, nb_heads_mult:float, 
            vocab_size:int, effective_sequence_size:int, 
            window_pattern:_WPattern)->tuple[GPT, GPTConfig]:
        """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
        ### cette fonction a été adaptée depuis 
        #   la demo de nonoChat, les constantes y sont issues aussi
        # Model dim is nudged up to nearest multiple of head_dim for clean division
        # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
        aspect_ratio = 10.5 * nb_heads_mult
        max_seq_len = 2 * effective_sequence_size 
        # -> allow big speed boost when using < half of `max_seq_len`
        base_dim = depth * aspect_ratio
        model_dim = int(((base_dim + head_dim - 1) // head_dim) * head_dim)
        num_heads = int(model_dim // head_dim)
        config = GPTConfig(
            sequence_len=max_seq_len, vocab_size=vocab_size,
            n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
            window_pattern="".join(window_pattern))
        with torch.device("meta"):
            model_meta = GPT(config)
        return model_meta, config

    @property
    def vocab_size(self)->int:
        return self.tokenizer.get_vocab_size()
    
    @property
    def context_size(self)->int:
        """recomanded context size to use"""
        return (self.config.sequence_len // 2)

    @property
    def device(self)->torch.device:
        device = self.llm.get_device()
        assert isinstance(device, torch.device), \
            f"expected a {torch.device}, got: {type(device)}"
        return device