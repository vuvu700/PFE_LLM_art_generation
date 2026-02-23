from LLM.nanochat.tokenizer import HuggingFaceTokenizer
from tokenizers import Tokenizer as HFTokenizer
from pathlib import Path
import json

SPECIAL_TOKENS = ["<|output_start|>", "<|output_end|>"]

class Tokenizer(HuggingFaceTokenizer):
    
    @staticmethod
    def load(tokenizer_dir:Path):
        with open(tokenizer_dir) as f:
            data = json.load(f)
            print(f"Load tokenizer {data}")

    


