import pickle
import time
import enum
from typing import Literal
from pathlib import Path
import holo.files
from holo.profilers import Profiler, ProgressBar
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

class Verbose(enum.IntEnum):
    disabled = enum.auto()
    perEpoch = enum.auto()
    liveProgress = enum.auto()
    debug = enum.auto()

class Model():
    __slots__ = (
        "llm", "tokenizer", "optimizer", "historique", 
        "_save_manager", "_prof", "__nb_epoches_done", )
    
    __TOKENIZER_NAME = "tokenizer.json" # sauvgardé 1 fois, a la racine de l'IA
    __MODEL_NAME = "model.pkl" # save dans les versions
    __HISTORY_NAME = "history.json" # save dans les versions
    
    llm: GPT
    tokenizer: Tokenizer
    optimizer: DistMuonAdamW|MuonAdamW
    historique: Historique
    
    def __init__(
            self, save_name:str|None, tokenizer:Tokenizer|Path|None, 
            device:_Device, depth:int, head_dim:int,
            context_size:int, nb_heads_mult:float=1.0) -> None:
        """cree un model avec un nouveau LLM non entrainé\n
        args:
            `save_name`: le nom utilisé pour
                str -> utilise un nom choisit
                None -> genere un nom auto
            `tokenizer`: le tokenizer a utiliser
                Tokenizer -> utilise celui ci
                Path -> charge le tokenizer ciblé
                None -> prends celui dans `save_name` sinon leve une erreur
            `device`: la device ou on met le LLM
            `depth`, `head_dim`, `context_size`, `nb_heads_mult`: les params du LLM
        """
        ### --- save manager ---
        if save_name is None:
            save_name = holo.files.get_unique_name(
                MODELS_SAVE_DIRECTORY, prefix="modelAuto_", onlyNumbers=True)
            print(f"used auto save name: {save_name!r}")
        self._save_manager = SavedAiTree(MODELS_SAVE_DIRECTORY.joinpath(save_name))
        ### --- tokenizer ---
        if tokenizer is None:
            tokenizer = self.get_tokenizer_path()
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
        self.__nb_epoches_done = 0
        self._prof = Profiler([
            "iterDataloader", "splitBatch", "toDevice",
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

    def get_tokenizer_path(self)->Path:
        return self._save_manager.aiDirectory.joinpath(self.__TOKENIZER_NAME)

    @staticmethod
    def clear_empty_save_dir()->None:
        """supprime tout les dossier de Models vides"""
        for directory in MODELS_SAVE_DIRECTORY.iterdir():
            if not directory.is_dir():
                continue # => not a dir
            if holo.files.getSize(directory.as_posix()) == 0:
                directory.rmdir()

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
    
    @property
    def nb_epoches_done(self)->int:
        return self.__nb_epoches_done
    
    def save(self, versionName:str, replaceTokenizer:bool=True)->tuple[int, Path]:
        """sauvgarde le model dans son dossier et renvois la version cree\n
        `versionName`: le nom de la version crée\n
        `replaceTokenizer`: True -> overwrite le tokenizer sauvgardé
        return: (ID de la version, chemain de la version)"""
        ID = self._save_manager.currentNextVersion
        directory = self._save_manager.createNewVersionFolder(versionName)
        # -> LLM & optimizer
        torch.save(
            obj={"llm": self.llm, "optimizer": self.optimizer},
            f=directory.joinpath(self.__MODEL_NAME))
        # -> tokenizer
        tokenizerPath = self.get_tokenizer_path()
        if replaceTokenizer or (not tokenizerPath.exists()):
            self.tokenizer.save(tokenizerPath)
        # -> Historique
        self.historique.save(directory.joinpath(self.__HISTORY_NAME))
        return (ID, directory)
    
    @staticmethod
    def load(ai_name:str, versionID:int)->"Model":
        """sauvgarde le model dans son dossier et renvois la version cree\n
        `ai_name`: le nom de la version crée
        `versionID`: le numero de la version a charger
        return: (ID de la version, chemain de la version)"""
        raise NotImplementedError("TODO: still in developement")
    
    
    def train(
            self, dataset:SVGDataset, batch_size:int,
            nbEpoches:int, timeLimite:float, verbose:Verbose)->None:
        nbEpoch_done: int = 0
        time_start: float = time.perf_counter()
        time_since_start = lambda: (time.perf_counter()-time_start)
        progress_epoches = ProgressBar.simpleConfig(
                nbSteps=nbEpoches, taskName="epoches", useEma=False, updateEvery=0.0)
        # eta_epoches = RemainingTime_mean(finalAmount=nbEpoches, start=True)
        # eta_time = RemainingTime_mean(finalAmount=timeLimite, start=True)
        
        while (nbEpoch_done < nbEpoches) and (time_since_start() < timeLimite):
            # => can do another epoch
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            accum = MetricsAccumulator(usage="train", topK=5)
            progress_batches = ProgressBar.simpleConfig(
                nbSteps=len(dataloader), taskName="batches", useEma=False)
            
            epochID = self.nb_epoches_done
            batch_iterator = iter(dataloader)
            nb_batch_done: int = 0
            epoch_start_time = time.perf_counter()
            if verbose >= Verbose.perEpoch:
                print(f"\nstarting epoch: {nbEpoch_done+1}")
            memStart = torch.cuda.memory.memory_reserved()
            
            while True:
                # clear the cache
                memCurr = torch.cuda.memory.memory_reserved()
                if (memCurr-memStart) > 2 * 1e9:  # using ..Go
                    torch.cuda.memory.empty_cache()
                # get the batch (not a for loop so we can mesure the duration of 'next')
                with self._prof.mesure("iterDataloader"):
                    try: datas = next(batch_iterator)
                    except StopIteration:
                        break
                # get the datas from the batch
                with self._prof.mesure("splitBatch"):
                    inputs: torch.Tensor = datas["tokens"]
                    targets: torch.Tensor = datas["targets"]
                    dtIndexes: list[int] = datas["datasetIndex"].tolist()
                    svgIndexes: list[int] = datas["svgIndex"].tolist()
                    chunckIndexes: list[int] = datas["chunckIndex"].tolist()
                    nbChars: int = sum([len(dataset.chunks[i].text) for i in dtIndexes])
                # send the tokens to the device
                with self._prof.mesure("toDevice"):
                    inputs = inputs.to(torch.int64).to(self.device)
                    targets = targets.to(torch.int64).to(self.device)
                # forward pass of the LLM
                with self._prof.mesure("forward"):
                    logits = self.llm.forward(idx=inputs, targets=None)
                # compute the loss and the metrics
                with self._prof.mesure("metrics&loss"):
                    loss = accum.batch_logits_metrics(
                        logits, targets, totalNbChars=nbChars)
                # backward pass and step
                with self._prof.mesure("zero_grad"):
                    self.optimizer.zero_grad()
                with self._prof.mesure("backward"):
                    loss.backward()
                with self._prof.mesure("step"):
                    self.optimizer.step()
                # log the infos
                if verbose >= Verbose.liveProgress:
                    progress_batches.step()
                nb_batch_done += 1
            
            torch.cuda.memory.empty_cache()
            # => epoch finished
            nbEpoch_done += 1
            self.__nb_epoches_done += 1
            epoch_duration = (time.perf_counter() - epoch_start_time)
            # compute the metrics
            metrics = accum.get_metrics()
            for name in metrics.keys():
                self.historique.add_metric(name, metrics[name], epoch_id=epochID)
            self.historique.add_metric("epoch_duration", epoch_duration, epoch_id=epochID)
            # infos post epoches
            if verbose >= Verbose.debug:
                prettyPrint(self._prof.pretty_totalTimes())
                prettyPrint(accum._prof.pretty_totalTimes())
            if verbose >= Verbose.perEpoch:
                progress_epoches.step()
                if not progress_epoches.estimator.isFinished():
                    print() # step don't produce a newline
                print(f"trained on: {nb_batch_done} batch ({len(dataset)} chuncks) in {prettyTime(epoch_duration)}")
                print(f" -> {nb_batch_done/epoch_duration:.2f} batch/sec | {len(dataset)/epoch_duration:.2f} chuncks/sec")
                gm = lambda metric: self.historique.get_metric_value(metric, epoch_id=epochID)
                print(f" -> CE: {gm('CE_train'):.4g} | PPL: {gm('PPL_train'):.4g} | top-1: {gm('TOP-1_train'):.2%}")
            