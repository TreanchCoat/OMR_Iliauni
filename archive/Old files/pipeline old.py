"""
pipeline.py — End-to-end OMR pipeline orchestrator.

Stages
------
    image
      ↓  staff_rectifier      — straighten curved/skewed staff lines
    rectified.png
      ↓  preprocessing        — detect staves, crop, binarize, remove lines
    ProcessedScore
      ↓  symbol_detector      — YOLO inference (full-rectified-page coords)
    PageDetections
      ↓  score_to_xml         — assemble MusicXML
    score.xml

Usage (Python)
--------------
    from pipeline import run_pipeline

    result = run_pipeline(
        image_path = 'page1.png',
        output_dir = 'out/',
        model_path = 'models/deepscores_crops_v1.pt',
    )

    # result is a dict with paths to all outputs:
    #   result['rectified_image']  — straightened PNG
    #   result['detections_json']  — symbol coordinates
    #   result['xml_file']         — MusicXML

Usage (CLI)
-----------
    python pipeline.py page1.png out/ models/deepscores_crops_v1.pt
"""

from __future__ import annotations

import sys
import os
sys.path.append(r'S:\omr')

import json
from pathlib import Path
from typing import Optional, Dict, Any

import cv2

import staff_rectifier
import preprocessing
import symbol_detector
import score_to_xml


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(image_path:        str,
                 output_dir:        str,
                 model_path:        str,
                 # Stage toggles
                 rectify:           bool  = True,
                 save_debug_images: bool  = False,
                 # Detection params
                 conf_threshold:    float = 0.25,
                 use_cleaned_image: bool  = True,
                 # XML params
                 instrument_name:   str   = 'Clarinet',
                 midi_program:      int   = 72,
                 divisions:         int   = 4,
                 ) -> Dict[str, Any]:
    """
    Run the full pipeline on a single score image.

    Parameters
    ----------
    image_path         Input score image (handwritten or printed).
    output_dir         Where intermediate + final files are written.
    model_path         Path to YOLO .pt weights.
    rectify            If True, run the staff-rectifier stage.  Set False
                       when the input is already known to have straight lines
                       (skips the homography step entirely).
    save_debug_images  If True, save staff-rectifier visualizations into
                       output_dir/debug_rectify/.
    conf_threshold     YOLO confidence threshold.
    use_cleaned_image  Run YOLO on the staff-line-removed image (True) or
                       the original color crop (False).
    instrument_name    Used for part naming in MusicXML.
    midi_program       GM program number (72 = clarinet).
    divisions          Divisions per quarter note in the output XML.

    Returns
    -------
    Dict with paths and in-memory results:
        rectified_image    str   — path to the rectified PNG
        detections_json    str   — path to the detection JSON
        xml_file           str   — path to the MusicXML
        page_detections    PageDetections object (in memory)
        processed_score    ProcessedScore object  (in memory)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f'Pipeline output directory: {out}')

    # ── Stage 1: Rectify ──────────────────────────────────────────────
    rectified_path = out / 'rectified.png'
    if rectify:
        print('\n[1/4] Rectifying staff lines …')
        debug_dir = out / 'debug_rectify' if save_debug_images else out / '_tmp_rectify'
        rectified_img = staff_rectifier.process_image(
            image_path     = image_path,
            output_dir     = str(debug_dir),
            visualize      = save_debug_images,
        )
        if rectified_img is None:
            raise RuntimeError('Staff rectification failed — no staves detected')
        cv2.imwrite(str(rectified_path), rectified_img)
        # Clean up the temp dir if we weren't keeping debug images
        if not save_debug_images:
            for f in debug_dir.glob('*'):
                f.unlink()
            debug_dir.rmdir()
    else:
        print('\n[1/4] Skipping rectification (using input image directly)')
        rectified_img = cv2.imread(image_path)
        if rectified_img is None:
            raise FileNotFoundError(f'Cannot read image: {image_path}')
        cv2.imwrite(str(rectified_path), rectified_img)
    print(f'      Rectified image → {rectified_path}')

    # ── Stage 2: Preprocess (staff analysis + cleanup) ────────────────
    print('\n[2/4] Detecting staves and removing staff lines …')
    processed = preprocessing.preprocess_image(str(rectified_path))
    if not processed.all_staves:
        raise RuntimeError('No staves detected in rectified image')

    # ── Stage 3: Symbol detection ─────────────────────────────────────
    print('\n[3/4] Running symbol detection …')
    detections = symbol_detector.detect_page(
        processed_score    = processed,
        model_path         = model_path,
        conf_threshold     = conf_threshold,
        use_cleaned_image  = use_cleaned_image,
    )
    detections_path = out / 'detections.json'
    symbol_detector.save_json(detections, str(detections_path))
    n_total = sum(sd.total_detections for sd in detections.all_staves)
    print(f'      {n_total} symbols detected across '
          f'{len(detections.all_staves)} staves')

    # ── Stage 4: MusicXML ─────────────────────────────────────────────
    print('\n[4/4] Building MusicXML …')
    xml_path = out / 'score.xml'
    score_to_xml.build_score_xml(
        processed_score = processed,
        page_detections = detections,
        output_path     = str(xml_path),
        instrument_name = instrument_name,
        midi_program    = midi_program,
        divisions       = divisions,
    )

    print('\nDone.')
    return {
        'rectified_image':  str(rectified_path),
        'detections_json':  str(detections_path),
        'xml_file':         str(xml_path),
        'page_detections':  detections,
        'processed_score':  processed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _print_usage():
    print('Usage:')
    print('  python pipeline.py <image> <output_dir> [model_path] [--debug]')
    print()
    print('Examples:')
    print('  python pipeline.py page1.png out/')
    print('  python pipeline.py page1.png out/ models/deepscores_crops_v1.pt --debug')


if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) < 2:
        _print_usage()
        sys.exit(1)

    image_path = args[0]
    output_dir = args[1]

    save_debug = '--debug' in args
    args = [a for a in args if a != '--debug']

    # Default model path: look in a 'models' folder next to pipeline.py
    _script_dir = Path(__file__).parent
    _default_model = _script_dir / 'models' / 'deepscores_crops_v1.pt'

    if len(args) > 2:
        model_path = args[2]
        # If given as a relative path, resolve from cwd
        model_path = str(Path(model_path).resolve())
    else:
        model_path = str(_default_model)

    if not Path(model_path).exists():
        print(f'ERROR: Model not found at: {model_path}')
        print('Pass the full path as the third argument:')
        print(f'  python pipeline.py {image_path} {output_dir} S:\\path\\to\\model.pt')
        sys.exit(1)

    result = run_pipeline(
        image_path        = image_path,
        output_dir        = output_dir,
        model_path        = model_path,
        save_debug_images = save_debug,
    )

    print('\n─── Outputs ───')
    print(f'  Rectified PNG : {result["rectified_image"]}')
    print(f'  Detections    : {result["detections_json"]}')
    print(f'  MusicXML      : {result["xml_file"]}')
