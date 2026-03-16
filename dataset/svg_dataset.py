import numpy
from typing import Callable, TypedDict
from pathlib import Path
import attrs
from lxml import etree  # type: ignore
import vpype 
import tempfile
import subprocess

from torch.utils.data import Dataset

from tokenizer_pfe.tokenizer_project import START_TOKEN, END_TOKEN


_Tokens = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int32]]
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
    tokensInput: _Tokens
    """tokens du chunck en input du LLM"""
    tokensOutput: _Tokens
    """tokens du chunck en output du LLM"""
    text: str
    """le text initial associé au tokens du chunk"""
    indexes: ChunckInfos
    """de que chunck dans quel svg il s'agit"""


class BatchDatas(TypedDict):
    tokens: _Tokens
    targets: _Tokens
    datasetIndex: int
    svgIndex: int
    chunckIndex: int


def clean_svg(svg: str) -> str:
    root = etree.fromstring(svg.encode('utf-8'), parser=parser)
    return etree.tostring(root, encoding='unicode', pretty_print=False)


def load_svg_samples(svg_dir: Path) -> list[SVGSample]:
    svg_samples: list[SVGSample] = []
    for svg_file in sorted(Path(svg_dir).glob('*.svg')):
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
        chunk = numpy.asarray(tokens[start: end+1], dtype=numpy.int32)
        assert chunk.ndim == 1
        chunks.append(chunk)
        i += 1
    return chunks

def svg_to_gcodes(svg_text: str, relative: bool = False) -> str:
    """ convertit un svg en gcode en utilisant vpype """
    profile = "gcode_relative" if relative else "gcode"
    # create temporary files for the svg input 
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False, mode="w", encoding="utf-8") as tmp_svg:
        tmp_svg.write(svg_text)
        tmp_svg_path = Path(tmp_svg.name)
    # create a temporary file for the gcode output
    with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as tmp_gcode:
        tmp_gcode_path = Path(tmp_gcode.name)
    try:
        # use vpype to convert the svg to gcode with the equivalent command line:
        # vpype read input.svg gwrite --profile gcode output.gcode
        result = subprocess.run(
            ["vpype", "read", str(tmp_svg_path), "gwrite", "--profile", profile,  str(tmp_gcode_path)],
            capture_output=True, text=True)
        if result.returncode != 0:  
            raise RuntimeError(f"vpype error: {result.stderr}")
        with open(tmp_gcode_path, 'r', encoding='utf-8') as f:
            return f.read()
    finally:
        tmp_svg_path.unlink(missing_ok=True)  # delete the temporary svg file
        tmp_gcode_path.unlink(missing_ok=True)  # delete the generated gcode
    return 


class SVGDataset(Dataset):
    def __init__(
        self,
        svg_dir: Path,
        context_size: int = 4096,
        tokenizer: Callable[[str], list[int]] = DEFAULT_TOKENIZER,
        decoder: Callable[[list[int]], str] = DEFAULT_DECODER,
        fillMissingTokens:bool=True,
        use_gcode: bool = True,
        use_relative_gcode: bool = False,   
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
            # convert the svg text to gcode if needed
            txt = sample.txt
            if use_gcode:
                txt = svg_to_gcodes(txt, relative=use_relative_gcode)
            # add the start and stop tokens
            # 2xStop => LLM learn that "Stop -> Stop"
            tokens = self.tokenizer("".join([
                START_TOKEN, txt, END_TOKEN, END_TOKEN]))
            svg_chunks = chunk_tokens(tokens, context_size)
            del tokens, sample
            for chunck_index, tokensInOut in enumerate(svg_chunks):
                tokensInOut_raw = tokensInOut
                # add padding at the end if needed (to size: context+1)
                nbMissingTokens: int = (self.context_size+1 - len(tokensInOut))
                assert nbMissingTokens >= 0, \
                    f"[BUG] unexpected chunck size: {len(tokensInOut)} ({self.context_size+1=})"
                if self.fillMissingTokens and (nbMissingTokens > 0):
                    # => fill the missing tokens with IGNORE_INDEX (=> will be ignored)
                    tokensInOut = numpy.concat([
                        tokensInOut, 
                        numpy.full((nbMissingTokens, ), IGNORE_INDEX, dtype=numpy.int32)
                    ], axis=0)
                del nbMissingTokens
                # save the chunck
                self.chunks.append(DatasetChunck(
                    tokensInput=tokensInOut[: -1], tokensOutput=tokensInOut[1: ],
                    text=self.decoder(tokensInOut_raw[: -1].tolist()), # no padding
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
        return BatchDatas(
            tokens=ch.tokensInput,
            targets=ch.tokensOutput,
            datasetIndex=indexes.datasetIndex,
            svgIndex=indexes.svgIndex,
            chunckIndex=indexes.chunckIndex)
