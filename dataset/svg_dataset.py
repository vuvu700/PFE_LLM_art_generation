import numpy
from typing import Callable, TypedDict
from pathlib import Path
import attrs
from lxml import etree  # type: ignore
import vpype
import tempfile
import subprocess
import re

from torch.utils.data import Dataset

from tokenizer_pfe.tokenizer_project import START_TOKEN, END_TOKEN

_Tokens = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int32]]
"""works like a list of tokens"""
IGNORE_INDEX = -1

parser = etree.XMLParser(remove_comments=True, remove_blank_text=True)
DEFAULT_TOKENIZER: Callable[[str], list[int]] = lambda svg_text: list(
    map(ord, svg_text)
)
DEFAULT_DECODER: Callable[[list[int]], str] = lambda tokens: "".join(map(chr, tokens))

# Normalization constants
DEFAULT_SVG_WIDTH = 900  # for testing, will replace by max for svg
DEFAULT_SVG_HEIGHT = 900  # for testing, will replace by max for svg
GCODE_WIDTH = 9.375
GCODE_HEIGHT = 9.375


class SVGSample:
    def __init__(self, txt: str, svg_file: Path):
        self.txt: str = txt
        self.svg_file: Path = svg_file

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"svg_file={self.svg_file}, txt_len={len(self.txt)})"
        )


@attrs.frozen
class ChunckInfos:
    datasetIndex: int
    """index du chunck dans le dataset"""
    svgIndex: int
    """index du svg"""
    chunckIndex: int
    """index du chunck dans le svg"""


@attrs.frozen
class DatasetChunck:
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
    root = etree.fromstring(svg.encode("utf-8"), parser=parser)
    return etree.tostring(root, encoding="unicode", pretty_print=False)


def _normalize(x, y, svg_max_x, svg_max_y):
    """normalize svg coordinates to gcode coordinates
    (0,0 at bottom left, max at top right)"""
    nx = round((x / svg_max_x) * GCODE_WIDTH, 4)
    ny = round((1 - y / svg_max_y) * GCODE_HEIGHT, 4)
    return nx, ny


def clean_gcode(output):
    """clean the output of the gcode generation to keep only the G/M code lines
    and convert svg <line> to G-code"""
    if isinstance(output, list):
        output = "".join(output)
    output = (
        output.replace("<|output_start|>", "")
        .replace("<|end_gcode|>", "")
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
    )
    output = re.sub(r"><", ">\n<", output)  # add newlines between tags
    output = re.sub(
        r"(<line)", r"\n\1", output
    )  # add newlines before line tags (in case they are not separated by >)

    all_x = re.findall(r'x[12]="?([\d\.\-]+)"?', output)
    all_y = re.findall(r'y[12]="?([\d\.\-]+)"?', output)
    valid_x = [float(v) for v in all_x if v.count(".") <= 1]
    valid_y = [float(v) for v in all_y if v.count(".") <= 1]
    svg_max_x = max(valid_x) if valid_x else DEFAULT_SVG_WIDTH
    svg_max_y = max(valid_y) if valid_y else DEFAULT_SVG_HEIGHT

    raw_lines = re.split(r"[\n\r]+", output)
    clean_lines = []
    skipped = 0
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        # keep valid G/M code lines
        if re.match(r"^(?:G|M)\d+", line):
            line = re.sub(r"\s+", " ", line)
            clean_lines.append(line)
            continue
        # convert SVG <line> to G-code
        if "<line" in line:
            x1 = re.search(r'x1="?([\d\.\-]+)"?', line)
            y1 = re.search(r'y1="?([\d\.\-]+)"?', line)
            x2 = re.search(r'x2="?([\d\.\-]+)"?', line)
            y2 = re.search(r'y2="?([\d\.\-]+)"?', line)

            if x1 and y1 and x2 and y2:
                try:
                    x1, y1 = float(x1.group(1)), float(y1.group(1))
                    x2, y2 = float(x2.group(1)), float(y2.group(1))
                    x1, y1 = _normalize(x1, y1, svg_max_x, svg_max_y)
                    x2, y2 = _normalize(x2, y2, svg_max_x, svg_max_y)
                    clean_lines.append(f"G00 X{x1} Y{y1}")
                    clean_lines.append(f"G01 X{x2} Y{y2}")
                except ValueError:
                    skipped += (
                        1  # to see how many it falty coordinates it removes (debug)
                    )
                    continue

    if skipped > 0:
        print(f"[clean_gcode] Dropped {skipped} malformed SVG lines")

    return "\n".join(clean_lines)


def load_svg_samples(svg_dir: Path) -> list[SVGSample]:
    svg_samples: list[SVGSample] = []
    for svg_file in sorted(Path(svg_dir).glob("*.svg")):
        with open(svg_file, "r", encoding="utf-8") as f:
            svg_content = f.read()
            cleaned_svg = clean_svg(svg_content)
            svg_samples.append(SVGSample(txt=cleaned_svg, svg_file=svg_file))
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
        chunk = numpy.asarray(tokens[start : end + 1], dtype=numpy.int32)
        assert chunk.ndim == 1
        chunks.append(chunk)
        i += 1
    return chunks


def svg_to_gcodes(svg_text: str, relative: bool = False) -> str:
    """convertit un svg en gcode en utilisant vpype"""
    profile = "gcode_relative" if relative else "gcode"
    # create temporary files for the svg input
    with tempfile.NamedTemporaryFile(
        suffix=".svg", delete=False, mode="w", encoding="utf-8"
    ) as tmp_svg:
        tmp_svg.write(svg_text)
        tmp_svg_path = Path(tmp_svg.name)
    # create a temporary file for the gcode output
    with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as tmp_gcode:
        tmp_gcode_path = Path(tmp_gcode.name)
    try:
        # use vpype to convert the svg to gcode with the equivalent command line:
        # vpype read input.svg gwrite --profile gcode output.gcode
        result = subprocess.run(
            [
                "vpype",
                "read",
                str(tmp_svg_path),
                "gwrite",
                "--profile",
                profile,
                str(tmp_gcode_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"vpype error: {result.stderr}")
        with open(tmp_gcode_path, "r", encoding="utf-8") as f:
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
        fillMissingTokens: bool = True,
        use_gcode: bool = False,
        use_relative_gcode: bool = False,
    ):
        assert context_size % 2 == 0, f"la contexte size doit absolument etre paire"
        self.context_size: int = context_size
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.fillMissingTokens: bool = fillMissingTokens
        self.chunks: list[DatasetChunck] = []

        self.samples = load_svg_samples(svg_dir)
        if use_gcode:
            # convert the svg text to gcode if needed
            for sample in self.samples:
                sample.txt = svg_to_gcodes(sample.txt, relative=use_relative_gcode)

        for svg_index, sample in enumerate(self.samples):
            txt = sample.txt
            # add the start and stop tokens
            # 2xStop => LLM learn that "Stop -> Stop"
            tokens = self.tokenizer("".join([START_TOKEN, txt, END_TOKEN, END_TOKEN]))
            svg_chunks = chunk_tokens(tokens, context_size)
            del tokens, sample
            for chunck_index, tokensInOut in enumerate(svg_chunks):
                tokensInOut_raw = tokensInOut
                # add padding at the end if needed (to size: context+1)
                nbMissingTokens: int = self.context_size + 1 - len(tokensInOut)
                assert (
                    nbMissingTokens >= 0
                ), f"[BUG] unexpected chunck size: {len(tokensInOut)} ({self.context_size+1=})"
                if self.fillMissingTokens and (nbMissingTokens > 0):
                    # => fill the missing tokens with IGNORE_INDEX (=> will be ignored)
                    tokensInOut = numpy.concat(
                        [
                            tokensInOut,
                            numpy.full(
                                (nbMissingTokens,), IGNORE_INDEX, dtype=numpy.int32
                            ),
                        ],
                        axis=0,
                    )
                del nbMissingTokens
                # save the chunck
                self.chunks.append(
                    DatasetChunck(
                        tokensInput=tokensInOut[:-1],
                        tokensOutput=tokensInOut[1:],
                        text=self.decoder(tokensInOut_raw[:-1].tolist()),  # no padding
                        indexes=ChunckInfos(
                            datasetIndex=len(self.chunks),
                            svgIndex=svg_index,
                            chunckIndex=chunck_index,
                        ),
                    )
                )

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
            chunckIndex=indexes.chunckIndex,
        )
