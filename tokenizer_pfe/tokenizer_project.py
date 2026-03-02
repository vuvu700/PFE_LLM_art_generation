from LLM.nanochat.tokenizer import HuggingFaceTokenizer
from tokenizers import Tokenizer as HFTokenizer
from pathlib import Path
import json

SPECIAL_TOKENS = ["<|output_start|>", "<|output_end|>"]


class Tokenizer(HuggingFaceTokenizer):

    @classmethod
    def from_directory(cls, tokenizer_dir: Path):
        raise NotImplementedError(f"we dont use this methode")

    def save(self, tokenizer_path: Path):
        tokenizer_path = tokenizer_path.with_suffix(".json")
        directory = tokenizer_path.parent
        assert directory.exists(), \
            FileNotFoundError(
                f"missing directory to save the tokenizer: {directory.as_posix()}")
        print(f"saving the tokenizer to: {tokenizer_path.as_posix()}")
        # save the tokenizer to disk
        self.tokenizer.save(tokenizer_path.as_posix(), pretty=False)

    @classmethod
    def load(cls, tokenizer_path: Path):
        tokenizer_path = tokenizer_path.with_suffix(".json")
        directory = tokenizer_path.parent
        assert directory.exists(), \
            FileNotFoundError(
                f"missing directory to load the tokenizer: {directory.as_posix()}")
        print(f"loading the tokenizer from: {tokenizer_path.as_posix()}")
        tokenizer = HFTokenizer.from_file(tokenizer_path.as_posix())
        return cls(tokenizer)
