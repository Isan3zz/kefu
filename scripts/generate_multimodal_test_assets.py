"""Generate three geometric PNG fixtures for engineering tests."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw


PALETTE = {'red': ('#bd242e', 'circle'), 'blue': ('#284bcc', 'square'), 'green': ('#207a42', 'triangle')}


def generate(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for label, (color, shape) in PALETTE.items():
        image = Image.new('RGB', (240, 180), color)
        draw = ImageDraw.Draw(image)
        if shape == 'circle':
            draw.ellipse((74, 40, 166, 132), fill='white')
        elif shape == 'square':
            draw.rectangle((74, 40, 166, 132), fill='white')
        else:
            draw.polygon([(120, 40), (74, 132), (166, 132)], fill='white')
        draw.text((16, 150), shape.upper(), fill='white')
        path = directory/(shape+'.png')
        image.save(path)
        paths[label] = path
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('tests/fixtures/multimodal'))
    for path in generate(parser.parse_args().output_dir).values():
        print(path)
