# U-Net Staff Detector — Setup & Training Guide

This walks through training a U-Net for staff line segmentation on the
CVC-MUSCIMA dataset (1000 handwritten music pages by 50 different writers).

The trained model plugs into the existing pipeline as the highest-priority
staff detector, with the YOLO and classical detectors as fallbacks.

---

## Why a U-Net?

Bounding-box detectors (YOLO) and morphology-based detectors give us *one
box* or *one curve* per staff. For thin, repeated structures like staff
lines on a handwritten page — variable thickness, broken segments, ink
bleed, paper texture — pixel-level segmentation is the dominant approach
in modern OMR research.

The model takes a 512×512 BGR input and produces a 512×512 probability
mask: every pixel gets a `P(staff line)` score. We then threshold and
post-process the mask into a list of staff dicts that the existing
rectifier can consume — no other code changes needed.

---

## Step 1: Install dependencies

```bash
pip install torch torchvision opencv-python numpy
pip install albumentations          # only needed for training
```

For GPU training, install the CUDA build of torch matching your driver:
https://pytorch.org/get-started/locally/

---

## Step 2: Download CVC-MUSCIMA

Go to the CVC-MUSCIMA database page:
**http://pages.cvc.uab.es/cvcmuscima/index_database.html**

Download the **Staff Removal set** (~1.9 GB):
**http://datasets.cvc.uab.es/muscima/CVCMUSCIMA_SR.zip**

Unzip it somewhere with at least 5 GB free. The structure should look like:

```
CVCMUSCIMA_SR/
  CvcMuscima-Distortions/
    ideal/                  ← clean originals (recommended baseline)
      w-01/
        image/   p001.png … p020.png   ← input images
        gt/      p001.png … p020.png   ← staff-only masks
        symbol/  p001.png … p020.png   ← staff-removed (not used here)
      w-02/  …  w-50/
    kanungo/                ← noise distortion variant
    rotated/                ← rotation distortion variant
    curvature/              ← page curl variant
    staffline-interruption/
    typeset-emulation/
    staffline-y-variation-v1/   staffline-y-variation-v2/
    staffline-thickness-ratio/
    staffline-thickness-variation-v1/  …-v2/
    whitespeckles/
```

Each writer has 20 pages. With 50 writers and 11 variants you get **11,000
training pairs** — far more than enough.

License note: CVC-MUSCIMA is **CC BY-NC-SA 4.0** (non-commercial research
only).

---

## Step 3: Build the manifest

```bash
python dataset_prep.py path/to/CVCMUSCIMA_SR
```

This walks the dataset, finds every `image/p###.png` ↔ `gt/p###.png`
pair, and writes `muscima_manifest.json` with train/val splits.

By default, validation holds out 5 random writers (~1100 pairs across all
distortion variants), so the model is evaluated on handwriting it never
saw during training.

To train on a subset (faster experiments, less variety):

```bash
# Just the clean originals: 50 writers × 20 pages = 1000 pairs
python dataset_prep.py path/to/CVCMUSCIMA_SR --variants ideal

# Clean + Kanungo noise: 2000 pairs
python dataset_prep.py path/to/CVCMUSCIMA_SR --variants ideal kanungo
```

---

## Step 4: Train

Default training (recommended for first run):

```bash
python staff_unet_train.py muscima_manifest.json
```

Default config:
- 30 epochs
- Batch size 2
- 512×512 resolution
- AdamW, lr=1e-4
- Random rotation, perspective, brightness, blur, noise augmentations
- Combined Dice + BCE loss

### On an NVIDIA RTX 3050 (4 GB)
~3 hours total. The "ideal" variant only is faster (~30 min) but gives a
less robust model. The full 11-variant dataset trains best.

### On an 8 GB+ GPU
Increase batch size:
```bash
python staff_unet_train.py muscima_manifest.json --batch-size 8
```

### CPU-only
Possible but slow. Reduce resolution to compensate:
```bash
python staff_unet_train.py muscima_manifest.json --size 256 --batch-size 1 --epochs 10
```

### Output

Training writes:
- `models/staff_unet.pth` — best checkpoint (highest validation IoU)
- `models/staff_unet.history.json` — per-epoch loss & IoU history

You should see validation IoU climb steadily. By epoch 5 it's typically
above 0.85; by epoch 30 above 0.95 on this dataset.

---

## Step 5: Verify

```bash
python staff_detector_unet.py path/to/score.png
```

Saves three files in the working directory:
- `unet_prob.png` — probability mask (grayscale)
- `unet_mask.png` — binarised mask
- `unet_staff_test.png` — original with detected lines/boxes overlaid

---

## Step 6: Use it

**No code changes required** — the pipeline now picks U-Net automatically
when `models/staff_unet.pth` exists. You'll see this in pipeline output:

```
      strategy unet          : 9 staves, consistency=0.94
      → using U-Net with 9 staves
```

If you want to compare detectors on a single image:

```python
from staff_rectifier import process_image

# U-Net (default)
process_image('score.png', 'out_unet/')

# Force YOLO
process_image('score.png', 'out_yolo/', force_strategy='yolo')

# Force classical
process_image('score.png', 'out_classical/', force_strategy='handwritten')
```

---

## Tuning

If the model misses faint stafflines:
```python
process_image('score.png', 'out/', unet_threshold=0.3)
```

If it merges noteheads into stafflines (false positives):
```python
process_image('score.png', 'out/', unet_threshold=0.7)
```

Default is 0.5.

---

## Troubleshooting

**"CUDA out of memory" during training**
Reduce batch size: `--batch-size 1`. Or input size: `--size 384`.

**Validation IoU plateaus around 0.5–0.7**
Usually means the dataset wasn't loaded properly. Inspect a few
`(image, mask)` pairs visually to make sure masks are aligned to images
and white = staff line (we invert white→1 internally; the source masks
are black-staff-on-white-background).

**Inference is slow**
The U-Net runs at 512×512 even for high-resolution input pages. For real-
time use, downsize the input before passing to `process_image()`. The
mask is upsampled back to original resolution after inference, so this
costs accuracy on very high-res pages but speeds up inference 3-5×.

**No CUDA GPU available**
Training is impractical without one. Two options:
1. Train on Google Colab (free T4 GPU, ~1 hour for the full dataset).
2. Skip training, use YOLO + classical fallbacks.
