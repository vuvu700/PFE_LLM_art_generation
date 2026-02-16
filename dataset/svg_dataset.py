from torch.utils.data import Dataset
from pathlib import Path
import re
from lxml import etree

parser = etree.XMLParser(remove_comments=True, remove_blank_text=True)

class SVGSample:
    def __init__(self, txt:str, svg_file:Path):
        self.txt: str = txt
        self.svg_file:Path = svg_file
        
    def __str__(self):
        return f"{self.__class__.__name__}(svg_file={self.svg_file}, txt_len={len(self.txt)})"
        

def clean_svg(svg:str)->str:
    #svg = svg.replace('\r\n', '\n').replace('\r', '\n')
    root = etree.fromstring(svg.encode('utf-8'), parser=parser)
    return etree.tostring(root, encoding='unicode', pretty_print=False)



def load_svg_samples(svg_dir:Path):
    svg_samples = []
    for svg_file in Path(svg_dir).glob('*.svg'):
        with open(svg_file, 'r', encoding='utf-8') as f:
            svg_content = f.read()
            cleaned_svg = clean_svg(svg_content)
            svg_samples.append(SVGSample(txt=cleaned_svg, svg_file=svg_file))
    return svg_samples



class SVGDataset(Dataset):
    def __init__(self, svg_dir:Path):
        self.samples: list[SVGSample] = load_svg_samples(svg_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx:int)->str:
        return self.samples[idx].txt