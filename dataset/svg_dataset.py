from torch.utils.data import Dataset
from pathlib import Path
import re
from lxml import etree


class SVGSample:
    def __init__(self, txt, svg_file):
        self.txt = txt
        self.svg_file = svg_file
        
    def __str__(self):
        return f"SVGSample(svg_file={self.svg_file}, txt_len={len(self.txt)})"
        

def clean_svg(svg):
    svg = svg.replace('\r\n', '\n').replace('\r', '\n')
    parser = etree.XMLParser(remove_comments=True, remove_blank_text=True)
    root = etree.fromstring(svg.encode('utf-8'), parser=parser)
    return etree.tostring(root, encoding='unicode', pretty_print=False)



def load_svg_samples(svg_dir):
    svg_samples = []
    for svg_file in Path(svg_dir).glob('*.svg'):
        with open(svg_file, 'r', encoding='utf-8') as f:
            svg_content = f.read()
            cleaned_svg = clean_svg(svg_content)
            svg_samples.append(SVGSample(txt=cleaned_svg, svg_file=svg_file))
    return svg_samples



class SVGDataset(Dataset):
    def __init__(self, svg_dir):
        self.samples = load_svg_samples(svg_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx].txt