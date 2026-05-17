"""
preprocess_for_training.py — Batch preprocessing for YOLO training prep.

Walks a folder of score PNGs, runs the standard preprocessing pipeline on
each one (rectify → staff detect → crop + clean), and writes one PNG per
detected staff into an output folder.  The cleaned (staff-line-removed)
crop is what the OMR's YOLO sees at inference time, so that's what we
hand-label for training.

This script INTENTIONALLY stops after preprocessing — no symbol detection,
no XML generation.  The goal is purely "give me staff crops I can label".

Usage
-----
    python preprocess_for_training.py INPUT_DIR OUTPUT_DIR [options]

Examples::

    # Process every .png in S:\dataset\pages
    python preprocess_for_training.py S:\dataset\pages out\staves

    # Quick benchmark on the first 20 pages
    python preprocess_for_training.py S:\dataset\pages out\staves --limit 20

    # Resume after a crash (skips pages already in the manifest)
    python preprocess_for_training.py S:\dataset\pages out\staves --resume

    # Run multiple CPU workers (keep workers ≤ logical cores − 1)
    python preprocess_for_training.py S:\dataset\pages out\staves --workers 4

Output layout
-------------
::

    OUTPUT_DIR/
        images/             cleaned (binary, staff-lines-removed) PNGs,
                            one per staff — this is what you label.
            <stem>_s01.png
            <stem>_s02.png
            ...
        raw_crops/          (optional, --save-raw) the original colour
                            crop of each staff, for reference.
        rectified/          (optional, --save-rectified) the full
                            page after rectification.
        manifest.csv        one row per staff crop with metadata:
                              source_page, staff_index, part_id,
                              staff_in_part, top_y, bot_y, left_x,
                              right_x, line_spacing, crop_path
        errors.log          pages that failed to preprocess.
        timing.log          per-page wall-clock so you can read off ETA.

Notes
-----
* The U-Net is the slowest single stage.  GPU is recommended.  If
  ``torch.cuda.is_available()`` is False the script falls back to CPU
  but you should expect ~30 s/page instead of ~7 s/page.
* ``--workers`` spawns PROCESS-level workers.  The U-Net is loaded once
  per process; with multiple workers each one pins a copy on the GPU.
  Memory permitting (3050 = 4 GB) 1–2 workers is usually optimal; if
  you see OOM, drop to 1.  CPU stages run inside each worker so the
  parallelism helps the postprocessing too.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

# Bootstrap: add <project>/src to sys.path and load .env.  This script
# lives at <project>/scripts/, so its grandparent is the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
import env_loader  # noqa: E402

import cv2  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Per-page worker
# ─────────────────────────────────────────────────────────────────────────────

def _process_page(args: Tuple[str, str, dict]) -> dict:
    """
    Worker function (runs in a child process).  Returns a result dict
    that the parent collects via the executor.
    """
    image_path, output_dir_str, opts = args
    output_dir = Path(output_dir_str)
    stem = Path(image_path).stem
    t0 = time.perf_counter()
    record: dict = {
        'page':       image_path,
        'stem':       stem,
        'ok':         False,
        'n_staves':   0,
        'staves':     [],
        'error':      None,
        'timing':     {},
    }

    try:
        # ProcessPoolExecutor child processes get a fresh interpreter on
        # Windows ("spawn"), so they don't inherit the parent's sys.path.
        # Re-bootstrap the project layout here.
        import sys as _sys
        _proj_root = Path(__file__).resolve().parent.parent
        _src = str(_proj_root / 'src')
        if _src not in _sys.path:
            _sys.path.insert(0, _src)

        # Import lazily so the parent process doesn't pay model-load cost.
        import env_loader  # noqa: F401
        import staff_rectifier
        import preprocessing

        # Stage 1: Rectify (optional, controlled by opts['rectify'])
        t_rect_start = time.perf_counter()
        if opts['rectify']:
            rectified = staff_rectifier.process_image(
                image_path = image_path,
                output_dir = str(output_dir / '_tmp_rectify' / stem),
                visualize  = False,
            )
            if rectified is None:
                raise RuntimeError('Rectifier returned no staves')
            # Persist the rectified PNG to a temp path so preprocessing
            # can read it back (preprocessing takes a file path, not an
            # ndarray).
            rect_dir = output_dir / 'rectified' if opts['save_rectified'] \
                       else output_dir / '_tmp'
            rect_dir.mkdir(parents=True, exist_ok=True)
            rect_path = rect_dir / f'{stem}.png'
            cv2.imwrite(str(rect_path), rectified)
        else:
            rect_path = Path(image_path)
        record['timing']['rectify_s'] = time.perf_counter() - t_rect_start

        # Stage 2: staff detection + per-staff crop + staff-line removal
        t_pre_start = time.perf_counter()
        processed = preprocessing.preprocess_image(str(rect_path))
        record['timing']['preprocess_s'] = time.perf_counter() - t_pre_start

        if not processed.all_staves:
            raise RuntimeError('No staves detected after preprocessing')

        # Stage 3: save crops
        t_save_start = time.perf_counter()
        images_dir = output_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = output_dir / 'raw_crops'
        if opts['save_raw']:
            raw_dir.mkdir(parents=True, exist_ok=True)

        for staff_idx, ps in enumerate(processed.all_staves, start=1):
            crop_filename = f'{stem}_s{staff_idx:02d}.png'
            crop_path = images_dir / crop_filename
            cv2.imwrite(str(crop_path), ps.cleaned)
            if opts['save_raw']:
                raw_path = raw_dir / f'{stem}_s{staff_idx:02d}_raw.png'
                cv2.imwrite(str(raw_path), ps.crop)

            staff_data = ps.staff_data
            record['staves'].append({
                'source_page':     image_path,
                'stem':            stem,
                'staff_index':     staff_idx,
                'part_id':         getattr(staff_data, 'part_id', ''),
                'staff_in_part':   getattr(staff_data, 'staff_in_part', -1),
                'top_y':           int(staff_data.top_y),
                'bot_y':           int(staff_data.bot_y),
                'left_x':          int(staff_data.left_x),
                'right_x':         int(staff_data.right_x),
                'line_spacing':    float(staff_data.line_spacing),
                'crop_path':       str(crop_path.relative_to(output_dir)),
            })

        record['timing']['save_s'] = time.perf_counter() - t_save_start
        record['n_staves'] = len(processed.all_staves)
        record['ok']       = True

        # Clean up temp rectified file when we're not asked to keep it.
        if opts['rectify'] and not opts['save_rectified']:
            try:
                rect_path.unlink()
            except OSError:
                pass

    except Exception as e:                # noqa: BLE001
        record['error'] = f'{type(e).__name__}: {e}'
        record['traceback'] = traceback.format_exc()

    record['timing']['total_s'] = time.perf_counter() - t0
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

MANIFEST_FIELDS = [
    'source_page', 'stem', 'staff_index',
    'part_id', 'staff_in_part',
    'top_y', 'bot_y', 'left_x', 'right_x',
    'line_spacing', 'crop_path',
]


def _collect_inputs(input_dir: Path, exts: List[str]) -> List[Path]:
    pages = []
    for ext in exts:
        pages.extend(sorted(input_dir.rglob(f'*.{ext}')))
        pages.extend(sorted(input_dir.rglob(f'*.{ext.upper()}')))
    # Dedup, keep order
    seen = set()
    unique = []
    for p in pages:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


def _load_done_set(manifest_path: Path) -> set:
    """Return the set of source pages already in the manifest."""
    done = set()
    if not manifest_path.exists():
        return done
    with manifest_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row['source_page'])
    return done


def _format_eta(seconds: float) -> str:
    if seconds < 60:
        return f'{seconds:.0f}s'
    if seconds < 3600:
        return f'{seconds/60:.1f}m'
    return f'{seconds/3600:.2f}h'


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog='preprocess_for_training',
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('input_dir', help='Folder containing source PNGs '
                                      '(scanned recursively).')
    ap.add_argument('output_dir', help='Where to write the crops and '
                                       'manifest.')
    ap.add_argument('--ext', default='png,jpg,jpeg,tif,tiff',
                    help='Comma-separated image extensions to look for.')
    ap.add_argument('--limit', type=int, default=0,
                    help='Stop after this many pages (0 = no limit).')
    ap.add_argument('--workers', type=int, default=1,
                    help='Parallel worker processes (default 1; raise '
                         'cautiously — each loads its own U-Net into VRAM).')
    ap.add_argument('--no-rectify', dest='rectify', action='store_false',
                    help='Skip the rectifier (use when staves are already '
                         'straight).  Saves ~3-5 s/page.')
    ap.add_argument('--save-raw', action='store_true',
                    help='Also save the original colour crop of each '
                         'staff to OUTPUT_DIR/raw_crops/.')
    ap.add_argument('--save-rectified', action='store_true',
                    help='Keep the full rectified page in '
                         'OUTPUT_DIR/rectified/.')
    ap.add_argument('--resume', action='store_true',
                    help='Skip pages already present in manifest.csv.')
    args = ap.parse_args(argv)

    input_dir  = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f'Input dir not found: {input_dir}', file=sys.stderr)
        return 2

    exts = [e.strip().lower().lstrip('.') for e in args.ext.split(',') if e.strip()]
    pages = _collect_inputs(input_dir, exts)
    if not pages:
        print(f'No images found under {input_dir} '
              f'(extensions: {exts})', file=sys.stderr)
        return 1
    print(f'Discovered {len(pages)} candidate page(s).')

    manifest_path = output_dir / 'manifest.csv'
    errors_path   = output_dir / 'errors.log'
    timing_path   = output_dir / 'timing.log'

    done_pages: set = set()
    if args.resume:
        done_pages = _load_done_set(manifest_path)
        if done_pages:
            print(f'Resume mode: {len(done_pages)} pages already done; '
                  f'they will be skipped.')

    todo = [p for p in pages if str(p) not in done_pages]
    if args.limit:
        todo = todo[:args.limit]
    print(f'Will process {len(todo)} page(s).')

    if not todo:
        print('Nothing to do.')
        return 0

    # Open files in append mode so resume works naturally.
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    opts = {
        'rectify':         args.rectify,
        'save_raw':        args.save_raw,
        'save_rectified':  args.save_rectified,
    }
    worker_args = [(str(p), str(output_dir), opts) for p in todo]

    t_start = time.perf_counter()
    n_done = 0
    n_failed = 0
    total_staves = 0

    with manifest_path.open('a', newline='', encoding='utf-8') as mf, \
         errors_path.open('a', encoding='utf-8') as ef, \
         timing_path.open('a', encoding='utf-8') as tf:

        writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
            mf.flush()

        def _flush_record(record: dict) -> None:
            nonlocal n_done, n_failed, total_staves
            n_done += 1
            stem = record['stem']
            if record['ok']:
                for s in record['staves']:
                    writer.writerow({k: s[k] for k in MANIFEST_FIELDS})
                total_staves += record['n_staves']
                tf.write(f'{stem}\t{record["n_staves"]}\t'
                         f'{record["timing"].get("rectify_s", 0):.2f}\t'
                         f'{record["timing"].get("preprocess_s", 0):.2f}\t'
                         f'{record["timing"].get("save_s", 0):.2f}\t'
                         f'{record["timing"]["total_s"]:.2f}\n')
            else:
                n_failed += 1
                ef.write(f'{stem}\t{record["error"]}\n')
                ef.write(record.get('traceback', '') + '\n')
            mf.flush(); ef.flush(); tf.flush()

            elapsed = time.perf_counter() - t_start
            rate    = n_done / max(elapsed, 1e-9)
            eta_s   = (len(todo) - n_done) / max(rate, 1e-9)
            ok_mark = '✓' if record['ok'] else '✗'
            print(f'  [{n_done}/{len(todo)}] {ok_mark} {stem}  '
                  f'(staves={record["n_staves"]}, '
                  f't={record["timing"]["total_s"]:.1f}s)  '
                  f'rate={rate:.2f} pg/s  ETA={_format_eta(eta_s)}',
                  flush=True)

        if args.workers <= 1:
            # Serial mode — easier to debug and reuses the loaded U-Net.
            for wa in worker_args:
                record = _process_page(wa)
                _flush_record(record)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(_process_page, wa) for wa in worker_args]
                for f in as_completed(futures):
                    record = f.result()
                    _flush_record(record)

    elapsed = time.perf_counter() - t_start
    print()
    print(f'Finished {n_done} page(s) in {_format_eta(elapsed)}; '
          f'{n_failed} failures, {total_staves} staves saved.')
    if n_failed:
        print(f'  See {errors_path} for details.')
    print(f'  Manifest: {manifest_path}')
    print(f'  Crops   : {output_dir / "images"}')

    return 0 if n_failed == 0 else 3


if __name__ == '__main__':
    sys.exit(main())
