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
from pathlib import Path

# Bootstrap: add <project>/src to sys.path and load .env.
# env_loader lives at <project>/src/env_loader.py, so we walk up from this
# script (scripts/pipeline.py) into the project root and inject src/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
import env_loader  # noqa: E402  — loads .env and finishes the sys.path setup

import json
from typing import Optional, Dict, Any  # noqa: E402

import cv2  # noqa: E402

import staff_rectifier  # noqa: E402
import preprocessing  # noqa: E402
import symbol_detector  # noqa: E402
import bbox_refiner  # noqa: E402
import clef_splitter  # noqa: E402  — splits multi-clef "staves" between Stage 1 and Stage 2
import build_musicxml_v2  # noqa: E402  — chords, rests, fermatas, ornaments
import measure_recalculator  # noqa: E402  — re-runnable measure rebar


# ─────────────────────────────────────────────────────────────────────────────
# User-configurable defaults
# ─────────────────────────────────────────────────────────────────────────────
#
# Tweak these flags to change the default behaviour of `python pipeline.py`.
# Every value can still be overridden by passing keyword arguments to
# `run_pipeline()` programmatically.

# Stage 1: how to straighten / rectify the page.
#
#   True  – run the full multi-strategy rectifier (U-Net -> YOLO ->
#           classical) and homography-warp every staff to a flat
#           coordinate frame.  Recommended for handwritten or
#           photographed scores where stafflines may be curved or
#           skewed.  Slowest path.
#   False – SKIP homography entirely.  The page is used as-is and the
#           classical staffline detector in `score_analyzer` (used by
#           Stage 2) does the line-finding work.  Best for printed
#           scores with already-straight stafflines; much faster.
RECTIFY = True

# When True, dump three visualisation PNGs next to score.xml so you can
# eyeball what the network is "seeing":
#   - unet_mask.png       U-Net staffline probability heatmap (only
#                         emitted when RECTIFY=True and the U-Net was
#                         available)
#   - staves_detected.png rectified page with the per-staff bounding
#                         box drawn on top
#   - symbols_detected.png rectified page with every detected symbol
#                         box drawn on top (coloured by class)
# Independent of `save_labeled_crops`, which still produces one labelled
# crop per staff if enabled.
VISUALIZE_DETECTIONS = True

# When True, save the binarized-but-unprocessed crop of every staff
# (Otsu/global threshold, BEFORE staff-line removal) into
# `output/binarized/P1_staff01.png` etc.  Useful for diagnosing whether
# binarization itself or the staff-line removal step is responsible for
# image damage on a given input.
SAVE_BINARIZED_CROPS = True


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────


def _save_stage1_staves_on_input(input_path: str, out_dir: Path) -> int:
    """
    Re-run staff_rectifier.detect_staff_curves on the ORIGINAL input
    image and overlay every detected staff curve on it.  This lets you
    see how many staves the rectifier actually found *before* warping.

    Returns the number of staves Stage 1 detected (so we can log it).
    """
    img_color = cv2.imread(str(input_path))
    if img_color is None:
        raise FileNotFoundError(f'Cannot read input image: {input_path}')
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    staves = staff_rectifier.detect_staff_curves(img_gray, verbose=False)
    vis = staff_rectifier.draw_detection(img_color, staves)
    out_path = out_dir / 'stage1_staves_on_input.png'
    cv2.imwrite(str(out_path), vis)
    print(f'      Stage 1 staves     -> {out_path}  ({len(staves)} found)')
    return len(staves)


def _save_unet_mask(rectified_img, out_dir: Path) -> None:
    """
    Run the U-Net on the rectified page and dump the probability mask.

    Two files are produced:
        unet_mask.png        — binary staff-line mask (B&W).
        unet_prob_heatmap.png — coloured heatmap of the probability mask
                                (red = high confidence, blue = low).
    """
    import numpy as np
    from staff_detector_unet import predict_mask

    if rectified_img.ndim == 3:
        gray = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = rectified_img

    print('      Running U-Net for staff-mask visualization…')
    prob, binary = predict_mask(gray)

    cv2.imwrite(str(out_dir / 'unet_mask.png'), binary)

    # Convert prob (float 0..1) to a heatmap PNG
    heat = (np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8)
    heat_rgb = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_dir / 'unet_prob_heatmap.png'), heat_rgb)
    print(f'      U-Net mask         -> {out_dir / "unet_mask.png"}')
    print(f'      U-Net heatmap      -> {out_dir / "unet_prob_heatmap.png"}')


def _save_staves_overlay(rectified_img, processed, out_dir: Path) -> None:
    """
    Stage 2 visualisation: draw every staff that score_analyzer found on
    the RECTIFIED page (not the original).

    For each ProcessedStaff:
      - a green rectangle around the staff's vertical band
      - 5 red horizontal lines at each detected staff-line y-position
      - a label "P{part_id} #{staff_in_part}" in the top-left of the box

    Compare this with stage1_staves_on_input.png to see whether the
    rectifier-side detection and the score_analyzer-side detection agree
    on staff count and position.
    """
    vis = rectified_img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    h, w = vis.shape[:2]
    for ps in processed.all_staves:
        sd = ps.staff_data
        y1 = max(0, int(sd.top_y) - 2)
        y2 = min(h - 1, int(sd.bot_y) + 2)
        x1 = max(0, int(getattr(sd, 'left_x', 0)))
        x2 = min(w - 1, int(getattr(sd, 'right_x', w - 1)))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        for ly in sd.line_positions:
            cv2.line(vis, (x1, int(ly)), (x2, int(ly)), (40, 40, 220), 1)
        label = f'{sd.part_id} #{sd.staff_in_part}'
        cv2.putText(vis, label, (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)

    out_path = out_dir / 'stage2_staves_on_rectified.png'
    cv2.imwrite(str(out_path), vis)
    print(f'      Stage 2 staves     -> {out_path}  '
          f'({len(processed.all_staves)} found)')


def _save_symbols_overlay(rectified_img, page_detections, out_dir: Path) -> None:
    """
    Draw every detected symbol on the rectified page.

    Box colour is hash-based per class so the same class gets the same
    colour across runs (useful for spotting confusions visually).
    """
    import hashlib

    vis = rectified_img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    def _colour(cls: str):
        h = hashlib.md5(cls.encode()).digest()
        return int(h[0]) | 30, int(h[1]) | 30, int(h[2]) | 30

    for sd in page_detections.all_staves:
        crop_y1 = int(getattr(sd, 'crop_y1', 0))
        crop_x1 = int(getattr(sd, 'crop_x1', 0))
        for d in sd.detections:
            x1 = int(d.x1) + crop_x1
            y1 = int(d.y1) + crop_y1
            x2 = int(d.x2) + crop_x1
            y2 = int(d.y2) + crop_y1
            cv2.rectangle(vis, (x1, y1), (x2, y2), _colour(d.class_name), 1)

    out_path = out_dir / 'symbols_detected.png'
    cv2.imwrite(str(out_path), vis)
    print(f'      Symbol overlay     -> {out_path}')


def _save_binarized_crops(processed, out_dir: Path) -> None:
    """
    Save one binarized PNG per staff crop, BEFORE the staff-line
    removal step.  Lets you compare:

        binarized/PN_staffMM_binary.png   ← Otsu on the colour crop only
        labeled_crops/PN_staffMM_*.png    ← with staff lines removed

    A diff between the two tells you whether the loss happened during
    binarization or during line removal.
    """
    from staff_remover import binarize as _binarize
    bin_dir = out_dir / 'binarized'
    bin_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for ps in processed.all_staves:
        sd = ps.staff_data
        binary = _binarize(ps.crop)
        fname = f'{sd.part_id}_staff{sd.staff_in_part+1:02d}_binary.png'
        cv2.imwrite(str(bin_dir / fname), binary)
        saved += 1
    print(f'      Binarized crops    -> {bin_dir}  ({saved} files)')


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(image_path:         str,
                 output_dir:         str,
                 model_path:         str,
                 # Stage toggles  (defaults pulled from the module-level
                 # RECTIFY / VISUALIZE_DETECTIONS constants above so a
                 # single edit at the top of the file changes the CLI
                 # default behaviour).
                 rectify:            bool = RECTIFY,
                 visualize_detections: bool = VISUALIZE_DETECTIONS,
                 save_binarized_crops: bool = SAVE_BINARIZED_CROPS,
                 save_debug_images:  bool = False,
                 save_labeled_crops: bool = True,
                 # Detection params
                 conf_threshold:     float = 0.25,
                 use_cleaned_image:  bool  = True,
                 refine_boxes:       bool  = True,
                 # XML params
                 instrument_name:    str   = 'Clarinet',
                 midi_program:       int   = 72,
                 divisions:          int   = 4,
                 embed_coordinates:  bool  = True,
                 ) -> Dict[str, Any]:
    """
    Run the full pipeline on a single score image.

    Parameters
    ----------
    image_path         Input score image (handwritten or printed).
    output_dir         Where intermediate + final files are written.
    model_path         Path to YOLO .pt weights.
    rectify            If True, run the full multi-strategy staff
                       rectifier (U-Net -> YOLO -> classical) and warp
                       every staff to a flat coordinate frame.  Set
                       False to skip homography entirely; the classical
                       staffline detector in score_analyzer (Stage 2)
                       does the line-finding instead.  Faster, suitable
                       for printed scores with straight stafflines.
    visualize_detections
                       If True, save three page-level visualizations
                       next to score.xml: unet_mask.png (only when
                       rectify=True and the U-Net was available),
                       staves_detected.png, symbols_detected.png.
    save_debug_images  If True, save the per-stage staff-rectifier
                       visualizations into output_dir/debug_rectify/.
                       Independent of visualize_detections.
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

    # ── Stage 0: Stage 1 detection visualised on the ORIGINAL image ───
    # (Optional — only emitted when visualize_detections is on.  Helpful
    # for spotting cases where the rectifier finds N staves but Stage 2
    # then sees fewer on the warped output.)
    if visualize_detections:
        try:
            _save_stage1_staves_on_input(image_path, out)
        except Exception as e:
            print(f'      [warn] could not save Stage 1 staves overlay: {e}')

    # ── Stage 1: Rectify ──────────────────────────────────────────────
    rectified_path = out / 'rectified.png'
    used_unet = False
    stage1_staves_meta: Optional[list] = None
    if rectify:
        print('\n[1/4] Rectifying staff lines (U-Net / YOLO / classical)…')
        debug_dir = out / 'debug_rectify' if save_debug_images else out / '_tmp_rectify'
        rectified_img, stage1_staves_meta = staff_rectifier.process_image(
            image_path      = image_path,
            output_dir      = str(debug_dir),
            visualize       = save_debug_images,
            return_metadata = True,
        )
        if rectified_img is None:
            raise RuntimeError('Staff rectification failed — no staves detected')
        cv2.imwrite(str(rectified_path), rectified_img)
        # Clean up the temp dir if we weren't keeping debug images
        if not save_debug_images:
            for f in debug_dir.glob('*'):
                f.unlink()
            debug_dir.rmdir()
        # Note whether the U-Net was available — used below to decide
        # whether the U-Net visualization is meaningful.
        try:
            from staff_detector_unet import is_unet_available
            used_unet = bool(is_unet_available())
        except Exception:
            used_unet = False
    else:
        # rectify=False is the "static / printed-score" path.  The
        # classical staffline detector in score_analyzer (used by
        # Stage 2) will find the lines without homography.  Much faster.
        print('\n[1/4] Skipping homography (RECTIFY=False).  Stafflines '
              'will be found by the classical algorithm in Stage 2.')
        rectified_img = cv2.imread(image_path)
        if rectified_img is None:
            raise FileNotFoundError(f'Cannot read image: {image_path}')
        cv2.imwrite(str(rectified_path), rectified_img)
    print(f'      Rectified image -> {rectified_path}')

    # ── U-Net mask visualization (when rectified via the neural path) ─
    if visualize_detections and rectify and used_unet:
        try:
            _save_unet_mask(rectified_img, out)
        except Exception as e:
            print(f'      [warn] could not save U-Net mask visualization: {e}')

    # ── Stage 1b: clef-based stave splitting ─────────────────────────
    # Run YOLO on the rectified page once to find clefs.  If any Stage
    # 1 staff turns out to contain more than one clef (which means the
    # rectifier merged two side-by-side staves into one band), we split
    # the staff metadata at the clef midpoints so Stage 2 sees the real
    # number of staves.  The cost is one extra YOLO pass on the
    # rectified image — much cheaper than re-running the whole stack.
    if stage1_staves_meta:
        try:
            clefs = clef_splitter.detect_clefs_in_image(
                rectified_img, model_path)
            if clefs:
                before = len(stage1_staves_meta)
                stage1_staves_meta = clef_splitter.split_staves_by_clefs(
                    stage1_staves_meta, clefs,
                    img_w=rectified_img.shape[1])
                after = len(stage1_staves_meta)
                if after > before:
                    print(f'      clef-split: {before} -> {after} staves '
                          f'({len(clefs)} clefs detected)')
                if visualize_detections:
                    clef_splitter.annotate_image_with_clefs(
                        rectified_img, clefs,
                        out / 'clefs_detected.png')
        except Exception as e:
            print(f'      [warn] clef-based split skipped: {e}')

    # ── Stage 2: Preprocess (staff analysis + cleanup) ────────────────
    # Forward Stage 1's staff metadata so preprocessing.preprocess_image
    # can skip its classical re-detection.  When rectify=False (or the
    # rectifier returned no metadata) we fall back to the legacy
    # detect-from-scratch path on the input image.
    if stage1_staves_meta:
        print(f'\n[2/4] Using {len(stage1_staves_meta)} staves from Stage 1; '
              f'removing staff lines …')
    else:
        print('\n[2/4] Detecting staves and removing staff lines …')
    processed = preprocessing.preprocess_image(
        str(rectified_path),
        precomputed_staves=stage1_staves_meta,
    )
    if not processed.all_staves:
        raise RuntimeError('No staves detected in rectified image')

    if visualize_detections:
        try:
            _save_staves_overlay(rectified_img, processed, out)
        except Exception as e:
            print(f'      [warn] could not save staves overlay: {e}')

    # Binarized (but NOT staff-line-removed) crops, for diagnosing
    # whether binarization itself or the staff-line-removal step is
    # damaging printed input.  Same naming convention as labeled_crops.
    if save_binarized_crops:
        try:
            _save_binarized_crops(processed, out)
        except Exception as e:
            print(f'      [warn] could not save binarized crops: {e}')

    # ── Stage 3: Symbol detection ─────────────────────────────────────
    print('\n[3/4] Running symbol detection …')
    detections = symbol_detector.detect_page(
        processed_score    = processed,
        model_path         = model_path,
        conf_threshold     = conf_threshold,
        use_cleaned_image  = use_cleaned_image,
    )
    detections_path = out / 'detections.json'
    n_total = sum(sd.total_detections for sd in detections.all_staves)
    print(f'      {n_total} symbols detected across '
          f'{len(detections.all_staves)} staves')

    if visualize_detections:
        try:
            _save_symbols_overlay(rectified_img, detections, out)
        except Exception as e:
            print(f'      [warn] could not save symbols overlay: {e}')

    # Optional bbox refinement — tighten boxes via connected components
    # on the cleaned (staff-line-removed) image.  Highest-leverage step
    # for notehead pitch accuracy.
    if refine_boxes:
        print('      Refining bounding boxes …')
        bbox_refiner.refine_page(detections, processed, verbose=True)

    # Compute the id_map once.  save_json uses it for the JSON, and we
    # forward the same map into the XML builder so detections.json and
    # the omr-coordinates field in the XML share identical det_NNNN IDs.
    shared_id_map = symbol_detector.save_json(detections, str(detections_path))

    # Save labeled staff crops
    labeled_crops_dir = None
    if save_labeled_crops:
        labeled_crops_dir = out / 'labeled_crops'
        symbol_detector.visualize_detections(
            processed_score = processed,
            page_detections = detections,
            output_dir      = str(labeled_crops_dir),
        )

    # ── Stage 4: MusicXML ─────────────────────────────────────────────
    print('\n[4/4] Building MusicXML …')
    xml_path = out / 'score.xml'
    # divisions=None lets the v2 builder pick a value that exactly
    # represents every detected rhythmic glyph on the page (1 / 2 / 4 / 8
    # depending on whether 16th/32nd notes are present).
    build_musicxml_v2.build_score_xml_v2(
        processed_score   = processed,
        page_detections   = detections,
        output_path       = str(xml_path),
        instrument_name   = instrument_name,
        midi_program      = midi_program,
        divisions         = None if divisions in (None, 0) else divisions,
        embed_coordinates = embed_coordinates,
        id_map            = shared_id_map,
    )

    # ── Stage 4b: re-bar into measures using the detected time signature.
    # measure_recalculator reads <time> from the XML the v2 builder just
    # wrote and re-splits each part accordingly.  When no time signature
    # was detected, the call is a safe no-op (single measure per staff).
    # The recalculator is also exposed standalone so the user can re-run
    # it after manually fixing wrong notes:
    #     python measure_recalculator.py score.xml score.xml --time 3/4
    print('       re-barring measures from the detected <time> …')
    measure_recalculator.recalculate(str(xml_path), str(xml_path))

    print('\nDone.')
    return {
        'rectified_image':   str(rectified_path),
        'detections_json':   str(detections_path),
        'xml_file':          str(xml_path),
        'labeled_crops_dir': str(labeled_crops_dir) if labeled_crops_dir else None,
        'page_detections':   detections,
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

    # Default model path: look in a 'models' folder at the project root
    _script_dir = Path(__file__).parent
    _default_model = _script_dir.parent / 'models' / 'deepscores_crops_v1.pt'

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
    if result.get('labeled_crops_dir'):
        print(f'  Labeled crops : {result["labeled_crops_dir"]}')
