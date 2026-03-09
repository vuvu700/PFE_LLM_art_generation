import pickle
import time
from typing import Literal
from pathlib import Path
from holo.files import get_unique_name
from holo.profilers import Profiler, RemainingTime_mean, ProgressBar
from holo.prettyFormats import prettyTime, prettyPrint

import torch
from torch.utils.data import DataLoader

from .nanochat.gpt import GPT, GPTConfig, DistMuonAdamW, MuonAdamW
from .saveManager import SavedAiTree

from dataset.svg_dataset import SVGDataset
from paths_cfg import MODELS_SAVE_DIRECTORY
from tokenizer_pfe.tokenizer_project import Tokenizer
from metrics.historique import Historique
from metrics.metrics import MetricsAccumulator


_Device = Literal["cpu", "cuda"]
_WPatternChr = Literal["S", "L"]
_WPattern = tuple[_WPatternChr, ...]
_Verbose = Literal["disabled", "full", "per epoch"]

class Model():
    __slots__ = (
        "llm", "tokenizer", "optimizer", "historique", 
        "_save_manager", "_prof", )
    
    __TOKENIZER_NAME = "tokenizer.json" # sauvgardé 1 fois, a la racine de l'IA
    __LLM_NAME = "llm.pkl"
    __HISTORY_NAME = "history.json"
    
    llm: GPT
    tokenizer: Tokenizer
    optimizer: DistMuonAdamW|MuonAdamW
    historique: Historique
    _save_manager: SavedAiTree
    
    def __init__(
            self, save_name:str|None, tokenizer:Tokenizer|Path, 
            device:_Device, depth:int, head_dim:int,
            context_size:int, nb_heads_mult:float=1.0) -> None:
        ### --- save manager ---
        if save_name is None:
            save_name = get_unique_name(MODELS_SAVE_DIRECTORY, prefix="modelAuto_")
            print(f"used auto save name: {save_name!r}")
        self._save_manager = ... # SavedAiTree(MODELS_SAVE_DIRECTORY.joinpath(save_name))
        ### --- tokenizer ---
        if isinstance(tokenizer, Path):
            tokenizer = Tokenizer.load(tokenizer)
        self.tokenizer = tokenizer
        ### --- historique ---
        self.historique = Historique()
        ### --- LLM & optimizer---
        self.llm = self.__build_model_meta(
            depth=depth, head_dim=head_dim, nb_heads_mult=nb_heads_mult,
            vocab_size=self.vocab_size, effective_sequence_size=context_size,
            window_pattern=("S", "S", "S", "L"))
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
        ### --- others ---
        self._prof = Profiler([
            "verbose", "iterDataloader", "splitBatch", "toDevice",
            "forward", "metrics&loss", "zero_grad", "backward", "step"])
        
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
            window_pattern:_WPattern)->GPT:
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
        return model_meta

    @property
    def config(self)->GPTConfig:
        return self.llm.config

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
    
    def save(self, versionName:str)->tuple[int, Path]:
        """sauvgarde le model dans son dossier et renvois la version cree\n
        `versionName`: le nom de la version crée\n
        return: (ID de la version, chemain de la version)"""
        raise NotImplementedError("TODO: still in developement")
        ID = self._save_manager.currentNextVersion
        directory = self._save_manager.createNewVersionFolder(versionName)
        self._save_manager.aiDirectory.joinpath()
        # -> tokenizer
        # -> LLM & optimizer
        with open(directory.joinpath(self.__LLM_NAME)) as file:
            ...
        # -> Historique
        with open(directory.joinpath(self.__LLM_NAME)) as file:
            ... 
        
        return (ID, directory)
    
    @staticmethod
    def load(ai_name:str, versionID:int)->"Model":
        """... to fill"""
        raise NotImplementedError("TODO: still in developement")
    
    
    def train(
            self, dataset:SVGDataset, batch_size:int,
            nbEpoches:int, timeLimite:float, verbose:_Verbose)->None:
        nbEpoch_done: int = 0
        time_start: float = time.perf_counter()
        time_since_start = lambda: (time.perf_counter()-time_start)
        progress_epoches = ProgressBar.simpleConfig(
                nbSteps=nbEpoches, taskName="epoches", useEma=1.0,
                updateEvery=0.0, newLineWhenFinished=False)
        # eta_epoches = RemainingTime_mean(finalAmount=nbEpoches, start=True)
        # eta_time = RemainingTime_mean(finalAmount=timeLimite, start=True)
        
        while (nbEpoch_done < nbEpoches) and (time_since_start() < timeLimite):
            # => can do another epoch
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            accum = MetricsAccumulator(usage="train", topK=5)
            progress_batches = ProgressBar.simpleConfig(
                nbSteps=len(dataloader), taskName="batches", useEma=False)
            
            batch_iterator = iter(dataloader)
            nb_batch_done: int = 0
            epoch_start_time = time.perf_counter()
            print(f"starting epoch n°{nbEpoch_done+1}")
            
            while True:
                # clear teh cach
                torch.cuda.empty_cache()
                # get the batch
                with self._prof.mesure("iterDataloader"):
                    try: datas = next(batch_iterator)
                    except StopIteration:
                        break
                # get the datas from the batch 
                with self._prof.mesure("splitBatch"):
                    tokens: torch.Tensor = datas["tokens"].to(torch.int64)
                    assert isinstance(tokens, torch.Tensor)
                    dtIndexes: list[int] = datas["datasetIndex"].tolist()
                    svgIndex: list[int] = datas["svgIndex"].tolist()
                    chunckIndex: list[int] = datas["chunckIndex"].tolist()
                    nbChars: int = sum([len(dataset.chunks[i].text) for i in dtIndexes])
                # send the tokens to the device
                with self._prof.mesure("toDevice"):
                    tokens = tokens.to(self.device)
                # forward pass of the LLM
                with self._prof.mesure("forward"):
                    logits = self.llm.forward(idx=tokens, targets=None)
                # compute the loss and the metrics
                with self._prof.mesure("metrics&loss"):
                    loss = accum.batch_logits_metrics(
                        logits[:, : -1].contiguous(), 
                        tokens[:, 1:].contiguous(),
                        totalNbChars=nbChars)
                # backward pass and step
                with self._prof.mesure("zero_grad"):
                    self.optimizer.zero_grad()
                with self._prof.mesure("backward"):
                    loss.backward()
                with self._prof.mesure("step"):
                    self.optimizer.step()
                # log the infos
                with self._prof.mesure("verbose"):
                    progress_batches.step()
                nb_batch_done += 1
            
            # => epoch finished
            with self._prof.mesure("verbose"):
                progress_epoches.step(); print() # progress don't output a newline 
            nbEpoch_done += 1
            # log some infos (temporary)
            epoch_duration = (time.perf_counter()-epoch_start_time)
            print()
            print(f"performed: {nb_batch_done} batch ({len(dataset)} chuncks) in {prettyTime(epoch_duration)}")
            print(f" -> {nb_batch_done/epoch_duration:.2f} batch/sec | {len(dataset)/epoch_duration:.2f} chuncks/sec")
            res = accum.get_metrics()
            prettyPrint(self._prof.pretty_totalTimes())
            prettyPrint(accum._prof.pretty_totalTimes())
            prettyPrint(res)