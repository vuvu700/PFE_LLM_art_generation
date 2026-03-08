import numpy
from typing import Callable, TypedDict
from torch.utils.data import Dataset
from pathlib import Path
import attrs
from lxml import etree  # type: ignore

_Tokens = numpy.ndarray[tuple[int], numpy.dtype[numpy.int32]]
"""works like a list of tokens"""
IGNORE_INDEX = -1

parser = etree.XMLParser(remove_comments=True, remove_blank_text=True)
DEFAULT_TOKENIZER: Callable[[str], list[int]] = \
    lambda svg_text: list(map(ord, svg_text))
DEFAULT_DECODER: Callable[[list[int]], str] = \
    lambda tokens: "".join(map(chr, tokens))


class SVGSample:
    def __init__(self, txt: str, svg_file: Path):
        self.txt: str = txt
        self.svg_file: Path = svg_file

    def __str__(self):
        return f"{self.__class__.__name__}(" \
            f"svg_file={self.svg_file}, txt_len={len(self.txt)})"


@attrs.frozen
class ChunckInfos():
    datasetIndex: int
    """index du chunck dans le dataset"""
    svgIndex: int
    """index du svg"""
    chunckIndex: int
    """index du chunck dans le svg"""


@attrs.frozen
class DatasetChunck():
    tokens: _Tokens
    """tokens du chunck"""
    text: str
    """le text initial associé au tokens du chunk"""
    indexes: ChunckInfos
    """de que chunck dans quel svg il s'agit"""


class BatchDatas(TypedDict):
    tokens: _Tokens
    datasetIndex: int
    svgIndex: int
    chunckIndex: int


def clean_svg(svg: str) -> str:
    root = etree.fromstring(svg.encode('utf-8'), parser=parser)
    return etree.tostring(root, encoding='unicode', pretty_print=False)


def load_svg_samples(svg_dir: Path) -> list[SVGSample]:
    svg_samples: list[SVGSample] = []
    for svg_file in Path(svg_dir).glob('*.svg'):
        with open(svg_file, 'r', encoding='utf-8') as f:
            svg_content = f.read()
            cleaned_svg = clean_svg(svg_content)
            svg_samples.append(SVGSample(
                txt=cleaned_svg, svg_file=svg_file))
    return svg_samples


def chunk_tokens(tokens: list[int], context_size: int) -> list[_Tokens]:
    """return the list of chuncks to be used as samples in the dataset \
    from a tokenized single file `tokens`"""
    half = context_size // 2
    chunks: list[_Tokens] = []
    i = 0
    while True:
        start = i * half
        end = start + context_size
        if start >= len(tokens):
            break  # plus de tokens à couvrir
        chunk = numpy.asarray(tokens[start: end], dtype=numpy.int32)
        assert chunk.ndim == 1
        chunks.append(chunk)  # type: ignore -> alredy checked the dim
        i += 1
    return chunks


class SVGDataset(Dataset):
    def __init__(
        self,
        svg_dir: Path,
        context_size: int = 4096,
        tokenizer: Callable[[str], list[int]] = DEFAULT_TOKENIZER,
        decoder: Callable[[list[int]], str] = DEFAULT_DECODER,
        fillMissingTokens:bool=True,
    ):
        assert context_size % 2 == 0, \
            f"la contexte size doit absolument etre paire"
        self.context_size: int = context_size
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.fillMissingTokens: bool = fillMissingTokens
        self.chunks: list[DatasetChunck] = []

        self.samples = load_svg_samples(svg_dir)

        for svg_index, sample in enumerate(self.samples):
            tokens = self.tokenizer(sample.txt)
            svg_chunks = chunk_tokens(tokens, context_size)
            for chunck_index, token_chunk in enumerate(svg_chunks):
                self.chunks.append(DatasetChunck(
                    tokens=token_chunk,
                    text=self.decoder(token_chunk.tolist()),
                    indexes=ChunckInfos(
                        datasetIndex=len(self.chunks),
                        svgIndex=svg_index,
                        chunckIndex=chunck_index)
                ))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx: int) -> BatchDatas:
        ch = self.chunks[idx]
        indexes = ch.indexes
        tokens: _Tokens = ch.tokens
        nbMissingTokens: int = self.context_size - len(tokens) 
        assert nbMissingTokens >= 0, \
            f"[BUG] unexpected chunck size: {len(tokens)} ({self.context_size=})"
        if self.fillMissingTokens and (len(tokens) < self.context_size):
            # => fill the missing tokens with -1 (=> will be ignored)
            filler = numpy.full((nbMissingTokens, ), IGNORE_INDEX, dtype=numpy.int32)
            tokens = numpy.concat([tokens, filler], axis=0) # type: ignore
        return BatchDatas(
            tokens=tokens,
            datasetIndex=indexes.datasetIndex,
            svgIndex=indexes.svgIndex,
            chunckIndex=indexes.chunckIndex)
