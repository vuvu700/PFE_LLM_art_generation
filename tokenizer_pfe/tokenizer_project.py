import json
from pathlib import Path
from typing import overload

from LLM.nanochat.tokenizer import HuggingFaceTokenizer, HFTokenizer

START_TOKEN = "<|output_start|>"
END_TOKEN = "<|output_end|>"
SPECIAL_TOKENS = [START_TOKEN, END_TOKEN]


class Tokenizer(HuggingFaceTokenizer):
    """wrapper sur le tokenizer de nanochat pour l'adapter a nos besoins"""

    @overload
    def encode(self, text: str) -> list[int]: ...
    @overload
    def encode(self, text: list[str]) -> list[list[str]]: ...
    def encode(self, text: str | list[str]):
        return super().encode(text)

    @overload
    def decode(self, ids: int) -> str: ...
    @overload
    def decode(self, ids: list[int]) -> str: ...
    @overload
    def decode(self, ids: list[list[int]]) -> list[str]: ...
    def decode(self, ids: int | list[int] | list[list[int]]):
        return super().decode(ids)

    @classmethod
    def from_directory(cls, tokenizer_dir: Path):
        raise NotImplementedError(f"we dont use this methode")

    def save(self, tokenizer_path: Path):
        tokenizer_path = tokenizer_path.with_suffix(".json")
        directory = tokenizer_path.parent
        assert directory.exists(), FileNotFoundError(
            f"missing directory to save the tokenizer: {directory.as_posix()}"
        )
        print(f"saving the tokenizer to: {tokenizer_path.as_posix()}")
        # save the tokenizer to disk
        self.tokenizer.save(tokenizer_path.as_posix(), pretty=False)

    @classmethod
    def load(cls, tokenizer_path: Path):
        tokenizer_path = tokenizer_path.with_suffix(".json")
        directory = tokenizer_path.parent
        assert directory.exists(), FileNotFoundError(
            f"missing directory to load the tokenizer: {directory.as_posix()}"
        )
        print(f"loading the tokenizer from: {tokenizer_path.as_posix()}")
        tokenizer = HFTokenizer.from_file(tokenizer_path.as_posix())
        return cls(tokenizer)
