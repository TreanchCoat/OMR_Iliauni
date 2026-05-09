"""
process_score.py — Send a score image to the OMR API and save the outputs.

Usage
-----
    python process_score.py score.png
    python process_score.py score.png --out-dir results/
    python process_score.py score.png --url http://192.168.1.10:5000
"""

import argparse
import base64
import sys
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description='Send a score image to the OMR API.')
    parser.add_argument('image', help='Path to the score image (PNG, JPG, TIFF…)')
    parser.add_argument('--url', default='http://localhost:5000',
                        help='API base URL (default: http://localhost:5000)')
    parser.add_argument('--out-dir', default='.',
                        help='Where to save outputs (default: current directory)')
    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        print(f'Error: file not found: {image_path}')
        sys.exit(1)

    print(f'Sending {image_path.name} to {args.url}/process …')

    with open(image_path, 'rb') as f:
        response = requests.post(
            f'{args.url}/process',
            files={'image': (image_path.name, f, 'image/png')},
            timeout=300,
        )

    if not response.ok:
        print(f'Error {response.status_code}: {response.text[:300]}')
        sys.exit(1)

    data = response.json()

    # Save rectified image
    rect_path = out_dir / 'rectified.png'
    img_bytes = base64.b64decode(data['rectified_image_b64'])
    rect_path.write_bytes(img_bytes)
    print(f'Rectified image → {rect_path}')

    # Save MusicXML
    xml_path = out_dir / 'score.xml'
    xml_path.write_text(data['xml'], encoding='utf-8')
    print(f'MusicXML        → {xml_path}')


if __name__ == '__main__':
    main()
