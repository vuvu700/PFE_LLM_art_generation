import pickle
import uuid
import copy
import time
import enum
import attrs
from typing import Literal, Generator
from pathlib import Path
import holo.files
from holo.pointers import Pointer
from holo.profilers import Profiler, ProgressBar
from holo.prettyFormats import prettyTime, prettyPrint
from holo.parallel import Manager as ThreadsManager

import torch
from torch.utils.data import DataLoader

from .nanochat.gpt import GPT, GPTConfig, DistMuonAdamW, MuonAdamW
from .saveManager import SavedAiTree

from dataset.svg_dataset import SVGDataset, START_TOKEN, END_TOKEN
from paths_cfg import MODELS_SAVE_DIRECTORY
from tokenizer_pfe.tokenizer_project import Tokenizer
from metrics.historique import Historique
from metrics.metrics import MetricsAccumulator
from metrics.affichage import affiche_metrics


_Device = Literal["cpu", "cuda"]
_WPatternChr = Literal["S", "L"]
_WPattern = tuple[_WPatternChr, ...]


class Verbose(enum.IntEnum):
    disabled = enum.auto()
    perEpoch = enum.auto()
    liveProgress = enum.auto()
    debug = enum.auto()


@attrs.frozen
class GenerationStats():
    nb_tokens: int
    gen_time: float
    stop_reason: str


@attrs.define
class Wandb_run_config():
    run_name: str
    run_id: str

    @staticmethod
    def fromName(name: str) -> "Wandb_run_config":
        return Wandb_run_config(
            run_name=name,
            run_id=Wandb_run_config.genID())

    @staticmethod
    def genID() -> str:
        return str(uuid.uuid4())


class Model():
    __slots__ = (
        "llm", "tokenizer", "optimizer", "historique",
        "_save_manager", "_prof", "__nb_epoches_done",
        "_wandb_config", )

    __TOKENIZER_NAME = "tokenizer.json"  # sauvgardé 1 fois, a la racine de l'IA
    __MODEL_NAME = "model.pkl"  # save dans les versions
    __HISTORY_NAME = "history.json"  # save dans les versions
    __METADATAS_NAME = "metadatas.pkl"  # save dans les versions

    llm: GPT
    tokenizer: Tokenizer
    optimizer: DistMuonAdamW | MuonAdamW
    historique: Historique

    def __init__(
            self, save_name: str | None, tokenizer: Tokenizer | Path | None,
            device: _Device, depth: int, head_dim: int,
            context_size: int, nb_heads_mult: float = 1.0) -> None:
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
        # --- save manager ---
        if save_name is None:
            save_name = holo.files.get_unique_name(
                MODELS_SAVE_DIRECTORY, prefix="modelAuto_", onlyNumbers=True)
            print(f"used auto save name: {save_name!r}")
        self._save_manager = SavedAiTree(
            MODELS_SAVE_DIRECTORY.joinpath(save_name))
        # --- tokenizer ---
        if tokenizer is None:
            tokenizer = self.get_tokenizer_path()
        if isinstance(tokenizer, Path):
            tokenizer = Tokenizer.load(tokenizer)
        self.tokenizer = tokenizer
        # --- historique ---
        self.historique = Historique()
        # --- LLM & optimizer---
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
        self.llm = torch.compile(self.llm, dynamic=False)  # type: ignore
        # --- others ---
        self.__nb_epoches_done = 0
        self._wandb_config = Wandb_run_config.fromName(save_name)
        self._prof = Profiler([
            "iterDataloader", "splitBatch", "toDevice",
            "forward", "metrics&loss", "zero_grad", "backward", "step"])

    def show_infos(self) -> None:
        print(self.config)
        params = self.llm.num_scaling_params()
        params_Embed = (params['wte'] + params['value_embeds'])
        print(f"{params['total']:_d} total params "
              f"(embeding: {params_Embed:_d} | "
              f"last layer: {params['lm_head']:_d} | "
              f"transformer: {params['transformer_matrices']:_d})")
        print(
            f"on device: {self.device!r}, with effective context_size: {self.context_size}")

    @staticmethod
    def __build_model_meta(
            depth: int, head_dim: int, nb_heads_mult: float,
            vocab_size: int, effective_sequence_size: int,
            window_pattern: _WPattern) -> GPT:
        """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
        # cette fonction a été adaptée depuis
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

    def get_tokenizer_path(self) -> Path:
        return self._save_manager.aiDirectory.joinpath(self.__TOKENIZER_NAME)

    @staticmethod
    def clear_empty_save_dir() -> None:
        """supprime tout les dossier de Models vides"""
        for directory in MODELS_SAVE_DIRECTORY.iterdir():
            if not directory.is_dir():
                continue  # => not a dir
            if holo.files.getSize(directory.as_posix()) == 0:
                directory.rmdir()

    @property
    def config(self) -> GPTConfig:
        return self.llm.config

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def context_size(self) -> int:
        """recomanded context size to use"""
        return (self.config.sequence_len // 2)

    @property
    def device(self) -> torch.device:
        device = self.llm.get_device()
        assert isinstance(device, torch.device), \
            f"expected a {torch.device}, got: {type(device)}"
        return device

    @property
    def nb_epoches_done(self) -> int:
        return self.__nb_epoches_done

    def wandb_show_metrics(self, join: bool) -> None:
        """affiche les metrics du model sur wandb\n
        si la run n'existe pas, en crée une nouvelle, sinon update celle deja existante\n
        ATTENTION: la fonction s'execute de façon asyncrone donc l'update peut metre du temps \
            néanmoins l'historique du model peut etre modifié sans risque pendent ce temps la\n
        args:
            `join`: True -> attend la fin de l'affichage avant de return
                False -> return imediatement aprés avoir commencé l'affichage (instantané)"""
        # TODO: decomenter ce code une fois que affiche_metrics est implementé
        manager = ThreadsManager(nbWorkers=1, startPaused=False)
        manager.addWork(
        func=affiche_metrics,
        historique=copy.deepcopy(self.historique),
        run_name=self._wandb_config.run_name,
        run_ID=self._wandb_config.run_id) # type: ignore
        if join is True:
           manager.join()

    def save(self, versionName: str, replaceTokenizer: bool = True) -> tuple[int, Path]:
        """sauvgarde le model dans son dossier et renvois la version cree\n
        `versionName`: le nom de la version crée\n
        `replaceTokenizer`: True -> overwrite le tokenizer sauvgardé
        return: (ID de la version, chemain de la version)"""
        ID = self._save_manager.currentNextVersion
        directory = self._save_manager.createNewVersionFolder(versionName)
        # -> LLM & optimizer
        torch.save(
            obj={
                "llm": self.llm.state_dict(),
                "optimizer": self.optimizer.state_dict()},
            f=directory.joinpath(self.__MODEL_NAME))
        # -> tokenizer
        tokenizerPath = self.get_tokenizer_path()
        if replaceTokenizer or (not tokenizerPath.exists()):
            self.tokenizer.save(tokenizerPath)
        # -> Historique
        self.historique.save(directory.joinpath(self.__HISTORY_NAME))
        # -> metadatas
        with open(directory.joinpath(self.__METADATAS_NAME), mode="wb") as file:
            metadatas = {
                "prof": self._prof,
                "nb_epoch_done": self.__nb_epoches_done,
                "llm_config": self.config}
            pickle.dump(metadatas, file)
        return (ID, directory)

    @staticmethod
    def load(ai_name: str, versionID: int, device: torch.device) -> "Model":
        """sauvgarde le model dans son dossier et renvois la version cree\n
        `ai_name`: le nom de la version crée
        `versionID`: le numero de la version a charger
        return: (ID de la version, chemain de la version)"""
        AI_Folder = MODELS_SAVE_DIRECTORY.joinpath(ai_name)
        assert AI_Folder.exists(), \
            f"there is AI directory named: {ai_name!r} inside: {MODELS_SAVE_DIRECTORY.as_posix()}"
        tree = SavedAiTree(AI_Folder)
        version_dir = tree.getVersionDirectory(versionID)
        model = object.__new__(Model)
        model._save_manager = tree
        # -> tokenizer
        model.tokenizer = Tokenizer.load(
            AI_Folder.joinpath(Model.__TOKENIZER_NAME))
        # -> metadatas
        with open(version_dir.joinpath(Model.__METADATAS_NAME), mode="rb") as file:
            metadatas: dict = pickle.load(file)
            model._prof = metadatas["prof"]
            model.__nb_epoches_done = metadatas["nb_epoch_done"]
            config_datas = metadatas["llm_config"]
        # generate a new wandb ID to avoid any conflicts
        model._wandb_config = Wandb_run_config.fromName(ai_name)
        # -> history
        model.historique = Historique.load(
            version_dir.joinpath(Model.__HISTORY_NAME))
        # -> LLM & optimizer
        model_data: dict = torch.load(
            version_dir.joinpath(Model.__MODEL_NAME), map_location=device)
        # -> rebuild the model
        model.__rebuild_LLM(
            llm_datas=model_data["llm"],
            optim_datas=model_data["optimizer"],
            config=config_datas,
            device=device)
        return model

    def __rebuild_LLM(
            self, llm_datas: dict, optim_datas: dict,
            config: GPTConfig, device: torch.device) -> None:
        """utilitaire a appeler aprés avoir load un model\n
        principalement repris de nanochat.checkpoint_manager.build_model"""
        assert isinstance(config, GPTConfig)
        if device.type in {"cpu", "mps"}:
            # Convert bfloat16 tensors to float for CPU inference
            llm_datas = {
                k: v.float() if v.dtype == torch.bfloat16 else v
                for k, v in llm_datas.items()}
        llm_datas = {k.removeprefix("_orig_mod."): v
                     for k, v in llm_datas.items()}
        with torch.device("meta"):
            llm = GPT(config)
        # Load the model state
        llm.to_empty(device=device)
        # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
        llm.init_weights()
        llm.load_state_dict(llm_datas, strict=True, assign=True)
        optim = llm.setup_optimizer()
        optim.load_state_dict(optim_datas)
        # set the llm and optimizer on self
        self.llm = torch.compile(llm, dynamic=False)  # type: ignore
        self.optimizer = optim

    def train(
            self, dataset: SVGDataset, batch_size: int,
            nbEpoches: int, timeLimite: float, verbose: Verbose) -> None:
        nbEpoch_done: int = 0
        time_start: float = time.perf_counter()
        def time_since_start(): return (time.perf_counter()-time_start)
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
                    try:
                        datas = next(batch_iterator)
                    except StopIteration:
                        break
                # get the datas from the batch
                with self._prof.mesure("splitBatch"):
                    inputs: torch.Tensor = datas["tokens"]
                    targets: torch.Tensor = datas["targets"]
                    dtIndexes: list[int] = datas["datasetIndex"].tolist()
                    svgIndexes: list[int] = datas["svgIndex"].tolist()
                    chunckIndexes: list[int] = datas["chunckIndex"].tolist()
                    nbChars: int = sum([len(dataset.chunks[i].text)
                                       for i in dtIndexes])
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
                self.historique.add_metric(
                    name, metrics[name], epoch_id=epochID)
            self.historique.add_metric(
                "epoch_duration", epoch_duration, epoch_id=epochID)
            self.wandb_show_metrics(join=False)
            self.save(f"checkpoint-{self.__nb_epoches_done}")
            # infos post epoches
            if verbose >= Verbose.debug:
                prettyPrint(self._prof.pretty_totalTimes())
                prettyPrint(accum._prof.pretty_totalTimes())
            if verbose >= Verbose.perEpoch:
                progress_epoches.step()
                if not progress_epoches.estimator.isFinished():
                    print()  # step don't produce a newline
                print(
                    f"trained on: {nb_batch_done} batch ({len(dataset)} chuncks) in {prettyTime(epoch_duration)}")
                print(
                    f" -> {nb_batch_done/epoch_duration:.2f} batch/sec | {len(dataset)/epoch_duration:.2f} chuncks/sec")

                def gm(metric): return self.historique.get_metric_value(
                    metric, epoch_id=epochID)
                print(
                    f" -> CE: {gm('CE_train'):.4g} | PPL: {gm('PPL_train'):.4g} | top-1: {gm('TOP-1_train'):.2%}")

    @torch.inference_mode()
    def __generate_internal(self, tokens: list[int], temperature: float, top_k: int | None):
        assert isinstance(tokens, list)
        device = self.device
        ids = torch.tensor([tokens], dtype=torch.long,
                           device=device)  # add batch dim
        while True:
            if ids.size(1) > self.context_size:
                # cut the context to the right size
                ids = ids[:, -self.context_size:]
            logits = self.llm.forward(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token

    @torch.no_grad
    def generate_flow(
            self, start: None | str | list[int], decode_batch: int,
            temperature: float, top_k: int | None,
            max_tokens: int | None, max_time: float | None,
            statsPtr: Pointer[GenerationStats] | None = None) -> Generator[str, None, None]:
        """genere les tokens suivants a la volée, \
            s'arrete tout seul a la fin d'un fichier ou aux limites données\n
        args:
            `start`: debut de la sequence a generer
                None -> sequence vide (juste un token START)
                str -> est converti en tokens
                list[int] -> les tokens directement
            `decode_batch`: le nombre de tokens accumulés avant de decoder
            `temperature` et `top_k`: utilisés pour le sampling des logits
            `max_tokens`: la limite de tokens a generer
                int -> nombre max a ne pas depasser
                None -> pas de limite de nombre de tokens
            `max_time`: la limite de temps de generation
                int -> nombre max de secondes a ne pas depasser
                None -> pas de limite de temps
        yield: les tokens deja decodés par batch
        return: the reason why it stoped
        """
        # setup the iterator and start sequence
        if start is None:
            start = START_TOKEN
        if isinstance(start, str):
            start = self.tokenizer.encode(start)
        tokens_generator = self.__generate_internal(
            start, temperature=temperature, top_k=top_k)
        # setup the vars
        if statsPtr is None:
            statsPtr = Pointer()
        reason: str = "[BUG] stoped for no reason !?"
        nb_tokens_gen: int = 0
        start_time: float = time.perf_counter()
        def time_since_start(): return (time.perf_counter() - start_time)
        tokens_buffer: list[int] = []
        # find the end token value
        _ = self.tokenizer.encode(END_TOKEN)
        assert len(_) == 1
        end_token_value: int = _[0]
        while True:
            if (max_tokens is not None) and (nb_tokens_gen >= max_tokens):
                reason = "reached max_tokens"
                break  # => generated enougth tokens
            if (max_time is not None) and (time_since_start() >= max_time):
                reason = "reached max_time"
                break  # => enougth time spent
            # => can predict another token
            new_token = int(next(tokens_generator))
            nb_tokens_gen += 1
            if new_token == end_token_value:
                reason = "reached END_TOKEN"
                break  # => reached the end of the generation
            tokens_buffer.append(new_token)
            if len(tokens_buffer) >= decode_batch:
                # => has generated enougth tokens to yield some text
                yield self.tokenizer.decode(tokens_buffer)
                tokens_buffer.clear()
        # => stoped generating (for a reason)
        if len(tokens_buffer) > 0:
            yield self.tokenizer.decode(tokens_buffer)
        statsPtr.value = GenerationStats(
            nb_tokens=nb_tokens_gen, gen_time=time_since_start(),
            stop_reason=reason)
