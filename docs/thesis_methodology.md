# Methodology — Master's Thesis Draft

**Working title:** A Modular Pipeline for Optical Music Recognition with
Application to Georgian Folk Polyphony

> This file is a scaffold for the Methodology chapter. Each section
> starts with a paragraph of text you can refine, followed by a
> `**TO EXPAND**` list of points that need your judgement / data / extra
> writing. Treat it as a checklist; bullet points marked
> `**TO EXPAND**` are prompts, not finished text.

---

## 1. Overview

The system implements a four-stage end-to-end pipeline that takes a
scanned or photographed page of musical notation as input and produces
a MusicXML document as output. The four stages are *page rectification*,
*staff analysis and cleaning*, *symbol detection*, and *score
reconstruction*. The pipeline is deliberately modular: each stage
exposes a stable data structure (rectified image → `ProcessedScore` →
`PageDetections` → MusicXML element tree) so individual stages can be
swapped, re-run, or replaced without disturbing the rest of the system.

```
PNG page  ──► [1] staff_rectifier      ──► rectified.png
              [2] preprocessing         ──► ProcessedScore
              [3] symbol_detector       ──► PageDetections
              [4] build_musicxml_v2     ──► score.xml
              [4½] measure_recalculator ──► score.xml (re-barred)
```

**TO EXPAND**
- Insert a labelled block diagram of the pipeline as Figure 1.
- Add one short paragraph per stage describing the *responsibility* of
  that stage in plain language (no implementation detail) — what goes
  in, what comes out, what assumptions are made.
- State the design principle explicitly: "every stage is independently
  testable; the final MusicXML can be re-generated from a saved
  `PageDetections` JSON without re-running detection."

---

## 2. Dataset

Two distinct datasets are used:

* **CVC-MUSCIMA** — 1 000 handwritten pages (50 writers × 20 pages),
  released by the Universitat Autònoma de Barcelona with per-pixel
  staff-line ground-truth masks (`gt/` folder). Used exclusively for
  training the staff-segmentation network (Stage 1). The corpus is
  augmented to 11 000 pairs via the distortion-variant subdirectories.
* **DeepScoresV2** — a synthetic large-scale dataset of typeset music
  symbols with bounding-box annotations across approximately 135
  classes. Used for training the symbol-detection network (Stage 3),
  via a pretrained weight checkpoint (`deepscores_crops_v1.pt`).
* **Georgian folk-music corpus** — 15 100 scanned pages from
  [TO EXPAND: source, time range, who collected, copyright status].
  This corpus is the application target and the source of the
  fine-tuning data for the symbol detector.

**TO EXPAND**
- A short paragraph on each dataset describing its license, citation,
  and any preprocessing you applied (resizing, format conversion,
  filtering of unusable pages).
- Provenance and characterisation of the Georgian corpus: number of
  unique compositions, page resolution distribution, handwritten vs.
  printed split, instrument distribution. A histogram of page sizes
  would make a good figure.
- Ethics / copyright statement on the Georgian corpus. Were
  permissions obtained? Is the corpus redistributable?
- A separate held-out *evaluation set* — typically 10–20 pages with
  human-edited reference MusicXML that the trained system never sees.

---

## 3. Stage 1 — Page Rectification (`staff_rectifier`)

Photographed scores and old print suffer from page curvature, skew, and
local warping. The rectifier reconstructs a flat coordinate frame in
which staff lines are straight and horizontal. Three strategies are
attempted in priority order:

1. **U-Net staff segmentation** — a 31 M-parameter U-Net trained on
   CVC-MUSCIMA produces a per-pixel staff-line probability mask. Staff
   centerlines are recovered from the mask by connected-components and
   row clustering, then a homography is fit per-staff to produce a
   straightened image.
2. **YOLO bounding-box detector** — falls back to the OLA pretrained
   model when the U-Net produces fewer staves than expected.
3. **Classical computer vision** — four parameter variants of Hough-line
   detection plus row-sum heuristics, used as a last resort for
   pathological inputs.

This cascading design reflects an empirical finding: the U-Net is the
most accurate detector when its input matches the training distribution
(printed scores at native resolution), but degrades on heavily curved
photographs where classical CV is more forgiving. By chaining the
strategies we recover correct staff detection on both clean prints and
challenging handwritten/photographed pages.

**TO EXPAND**
- Justify each strategy choice. Why three? Cite alternative approaches
  you considered and rejected (e.g., line-drawing GAN, end-to-end
  transformer dewarpers).
- Tiled-inference detail (this is a defensible technical contribution).
  Explain: U-Net was trained at 512×512; naive 2 550×4 200 → 512 resize
  destroys 1–2-px staff lines as sub-pixel artifacts; tiled inference
  at native resolution with 64-px overlap recovers correct detections.
- Include before/after figures: raw photograph → U-Net mask → recovered
  staves → rectified output.

---

## 4. Stage 2 — Staff Analysis & Cleaning (`preprocessing`)

Given a rectified page, this stage detects every staff, clusters the
staves into rows, assigns each staff to a part (musical voice), and
removes the staff lines from each crop so the symbol detector sees only
the symbols. Staff-line removal is implemented in classical CV
(morphological reconstruction along horizontal strokes plus row-sum
profile thresholding); a neural approach was deliberately avoided here
because the operation is geometric and deterministic.

The row clusterer groups staves into *systems* (horizontally aligned
groups) using their top-y coordinates with a tolerance of half the
median staff height. Within each system, staves are assigned to parts
left-to-right (for side-by-side systems) or top-to-bottom (when each
"row" contains a single staff — a vertically-stacked layout typical of
choral or quartet writing).

**TO EXPAND**
- Document the part-assignment logic explicitly with an example.
- Discuss layout edge cases that remain unsolved: pickup measures,
  bracket/brace grouping, multi-staff piano, mid-piece changes in part
  count. Position this as future work.
- Compare against alternative staff-line removal methods (e.g.,
  Cardoso-Sotelo "Staff removal with patch-based U-Nets") and justify
  the classical choice on grounds of (i) determinism, (ii) zero training
  data needed, (iii) sufficient accuracy.

---

## 5. Stage 3 — Symbol Detection (`symbol_detector` + `bbox_refiner`)

A YOLOv8 detector is run on each cleaned staff crop independently. The
choice of YOLO over a transformer-based detector (DETR, Deformable
DETR) was driven by inference cost: YOLO supports CPU inference at
useful speed, allowing the system to run on the target deployment
hardware (commodity laptops without dedicated GPUs).

After detection, a *bounding-box refinement* pass tightens notehead
boxes using connected-components on the cleaned image. The refiner
recenters each notehead box on the centroid of its underlying CC blob
while preserving the original box size — this corrects sub-pixel
offsets that bias the downstream pitch calculation.

**TO EXPAND**
- Class taxonomy: list the YOLO classes you use, mapped to MusicXML
  primitives. Include a table.
- Discuss class imbalance in DeepScoresV2 (e.g., `noteheadBlack` is
  ~10⁵ samples but `ornamentTrill` is ~50). Describe mitigation:
  per-class confidence thresholds, focal loss, oversampling.
- Out-of-vocabulary symbols: five melisma types (shake, acciaccatura,
  nachshlang, long appoggiatura, glissando) appear in Georgian folk
  music but not in DeepScoresV2. The pipeline includes XML-emission
  hooks (`add_ornament`, `add_grace_note`, `add_glissando`) so that
  once the user supplies labelled examples, the framework needs no
  further changes.
- Pitch calculation formula and its assumptions about clef geometry.
  Show the derivation: `steps = round((middle_line_y − note_y) /
  half_spacing)` then mapped through diatonic alphabet wrapping.

---

## 6. Stage 4 — Score Reconstruction (`build_musicxml_v2`)

The detection output is assembled into a MusicXML document by direct
construction of the XML element tree (lxml). Every detection becomes a
MusicXML element with a stable `id` attribute, and the full coordinate
list is embedded in `<identification><miscellaneous>` as JSON. This
makes every output XML self-contained: a downstream user-correction
application can locate any visual symbol from its XML element and
vice-versa without re-running detection.

Design decisions in this stage:

* **No barline detection.** Bar boundaries are derived from time
  signatures detected by YOLO (see `detect_all_time_signatures`); see
  Stage 4½.
* **Chord grouping.** Noteheads within ±10 px on the x-axis are
  collapsed into a single `<note>` chain with `<chord/>` markers,
  sorted top-to-bottom.
* **Configurable divisions.** The `<divisions>` value is chosen by
  scanning detected rhythmic glyphs: divisions=1 for halves/quarters,
  ramping up to 16 for 64th-note flags. This keeps every detected
  duration an integer multiple of `divisions`, so no floating-point
  arithmetic enters the durations.
* **All optional MusicXML fields are optional in the builder.** This
  is a deliberate deviation from prior MusicXML construction libraries
  (e.g., music21) that auto-fill defaults — the builder writes only
  what was explicitly detected, so the output reflects the actual
  evidence.

**TO EXPAND**
- Trade-off discussion: why not infer barlines? Because per-pixel
  barline detection on staff-line-removed images is error-prone, and
  the bar placement carries musical-content meaning that detection
  alone can't recover.
- Coordinate-preservation rationale: cite use cases for embedded
  coordinates (correction UI, dataset bootstrapping, debugging).
- MusicXML schema compliance: discuss the `<defaults> → <part-list> →
  <part>` ordering bug you encountered with music21 and how the
  schema-correct insertion helper (`_insert_top_level`) solves it.

---

## 7. Stage 4½ — Measure Recalculation (`measure_recalculator`)

Measure boundaries are recomputed in a separate, re-runnable script
rather than inline in the XML builder. This design supports an
interactive workflow: a user opens the OMR output in an editor, hand-
corrects a wrong note's duration, then re-runs the recalculator to
re-bar the score without re-running the entire OMR pipeline.

Two splitting policies were considered:

1. **Beat-counting** — accumulate durations and close a bar every
   `beats × divisions × (4/beat-type)` units. Pros: aligns with
   conventional notation. Cons: error-amplifying — a single
   mis-classified note duration cascades into wrong bar boundaries
   throughout the piece.
2. **Boundary-preserving** — open a new bar at every detected staff
   boundary, plus an extra bar whenever a new `<time>` signature is
   detected mid-staff. Pros: structural; resilient to duration errors;
   each output bar corresponds to a visual region the user can verify.
   Cons: bars may contain "the wrong number of beats" by conventional
   standards.

The implementation adopts policy (2) based on a downstream consumer
analysis: the final correction tool needs structural correspondence
between bars in the XML and visible regions on the page, not metric
correctness.

**TO EXPAND**
- Cite this as a methodological contribution: the pipeline is the
  first OMR system (verify this claim) to expose a re-runnable
  re-barring tool decoupled from detection.
- Describe the mid-piece time-signature handling: YOLO's
  `detect_all_time_signatures` returns every signature with its
  x-position; the v2 builder interleaves them with note events; the
  recalculator splits on each one.
- Include a worked example: staff with notes A B C D | E F | G H,
  showing how a mid-stream 2/4-to-3/4 change is preserved end to end.

---

## 8. Training Methodology

### 8.1 Staff segmentation (U-Net)

The U-Net is trained on CVC-MUSCIMA pages using a combined BCE+Dice
loss. The training run was performed on Vast.ai cloud instances (RTX
4090 / 5090); the final checkpoint reached IoU 0.85 on a held-out
validation split after 15 epochs at batch size 16. Training proceeded
without spatial augmentation — the underlying U-Net architecture and
the high diversity of CVC-MUSCIMA writers proved sufficient.

A critical methodological detail: the ground-truth masks in
CVC-MUSCIMA are encoded as `WHITE = staff line` on a `BLACK` background
(`mask >= 128`). An initial training run interpreted the polarity in
reverse (`mask < 128`), producing an apparent IoU of 0.96 from a
trivial "predict everything positive" prediction. The bug was caught
by computing the empirical pixel-density of the training masks and
finding 97% black pixels — the model was essentially memorising
"return 1.0 everywhere". Polarity fix and re-training brought the
honest IoU back to 0.65 → 0.85 over 15 epochs.

**TO EXPAND**
- A figure of the loss curve (BCE+Dice over epochs) and the validation
  IoU per epoch, both before and after the polarity fix. This is a
  good honest-debugging story for the thesis.
- Hyperparameters table: learning rate, optimizer, batch size, weight
  decay, augmentation policy (none), data-loading workers, epoch
  count, GPU spec, wall-clock cost.

### 8.2 Symbol detection (YOLOv8)

YOLOv8 is initialised from a DeepScoresV2-pretrained checkpoint. Two
fine-tuning regimes are evaluated:

* **Zero-shot** — the off-the-shelf DeepScoresV2 weights are used
  without any adaptation. This is the current state of the system; it
  achieves [TO MEASURE]% mAP on the Georgian held-out set.
* **Fine-tuned** — the DeepScoresV2 checkpoint is fine-tuned on
  approximately 15 100 pages of Georgian folk scores after manual
  labelling. Augmentation policy: random crops, brightness/contrast
  jitter (handwritten scores have variable exposure), small rotations
  (≤ 2°) within the rectified-page convention.

**TO EXPAND**
- mAP@0.5 and mAP@0.5:0.95 numbers per class for both regimes.
- A confusion matrix to highlight which symbols the model still
  confuses (handwritten flag vs. beam segment is a typical
  pathological case in OMR).
- A learning-rate ablation: warm-up schedule, multi-step decay vs.
  cosine.

### 8.3 Hardware and reproducibility

Local development hardware: RTX 3050 4 GB, Windows 10. Cloud training:
Vast.ai RTX 4090 / 5090 at approximately USD 0.31 / 0.37 per hour.
Preprocessing of the 15 100-page training corpus is estimated at
roughly 8–10 hours on a single 5090 with batched U-Net inference; this
is dominated by the connected-components and staff-line-removal stages
(CPU-bound) rather than the U-Net itself.

**TO EXPAND**
- Document the exact pip-freeze (or conda environment) used for each
  training and inference run. The thesis appendix should include this
  verbatim for reproducibility.
- Document seed handling. Were the runs deterministic? If not, state
  whether the reported numbers are single-run or averaged.

---

## 9. Evaluation Methodology

Three evaluation regimes are reported, in increasing order of strictness:

1. **Symbol-level detection accuracy** — mAP on the held-out symbol
   dataset. Measures stage 3 only.
2. **Per-element MusicXML diff** — the pipeline output is compared to
   a hand-corrected reference MusicXML, element by element. Reported
   as precision/recall for each MusicXML primitive (notehead, rest,
   clef, key, time, accidental, slur, tie, fermata, ornament).
3. **Listenability** — the resulting MusicXML is exported to MIDI via
   MuseScore and rated by a music-trained reviewer on a 5-point scale
   ([acceptable for sight-reading, acceptable with minor edits, …, not
   recoverable]). Reported as the distribution over the held-out set.

The third regime is the most application-relevant: the system is
intended for downstream use by performers, not as a benchmark, so the
correct metric is whether a musician can read the output.

**TO EXPAND**
- Define each metric precisely. For (2), state how note-equivalence is
  determined: same pitch (step+octave+alter), same `<duration>`, same
  `<voice>`?
- Compare against published baselines on a public dataset (e.g.,
  PrIMuS, MUSCIMA++) so the thesis has external calibration.
- Statistical significance: if you compare zero-shot vs. fine-tuned,
  report the confidence interval (bootstrap on pages, n=20+).

---

## 10. Limitations and Future Work

Items already identified during development as out-of-scope for this
thesis:

* Multi-page assembly (clef/key/time continuity across page breaks)
* Voice separation on polyphonic staves
* Beam-stack counting for distinguishing 16th from 8th notes
* Tuplet detection (triplets etc.)
* Cross-staff beaming (piano notation)
* Bracket/brace recognition for instrument grouping
* Pickup measures (anacrusis numbering)
* Cadenza / unmetered passages
* Five melisma classes absent from DeepScoresV2 (framework ready;
  labelling pending)
* Pitch calculation has been validated for treble, treble-8vb, bass,
  alto, tenor clefs but not for percussion/drum clefs.

**TO EXPAND**
- Briefly justify *why* each item was deferred (engineering scope,
  data availability, dependency on user manual labelling, …).
- Position the *correction UI* in this section: the OMR exposes the
  hooks (per-element IDs, coordinate JSON, re-runnable recalculator)
  but the UI itself is an external project beyond the thesis scope.

---

## 11. Reproducibility checklist

Following Pineau et al. (NeurIPS 2019) reproducibility guidance:

- [ ] Source code available at `https://github.com/TreanchCoat/OMR_Iliauni`
- [ ] Trained model checkpoints released (with Git LFS for the .pt
      files; see project README)
- [ ] Random seeds documented per training run
- [ ] Hyperparameters documented in a single config file (or table in
      the thesis appendix)
- [ ] Dataset version pins: CVC-MUSCIMA download URL, DeepScoresV2
      version, Georgian corpus access method
- [ ] Hardware spec for every reported number
- [ ] Wall-clock cost per training run

---

## 12. Notes for thesis writing

A few tactical pointers as you flesh this out:

* **Cite as you draft.** Every empirical claim ("YOLOv8 is faster than
  DETR on CPU", "tiled inference recovers staff lines lost to resizing")
  needs a citation or a backed-up experiment. Maintain a `references.bib`
  file from day one.
* **Make the design decisions defensible.** For each stage, you should
  be able to answer "why this and not the alternative?" in one sentence,
  with one citation. The current methodology already takes a position
  on roughly a dozen of these (U-Net not DETR for staff; YOLO not DETR
  for symbols; classical not neural for staff-line removal; boundary-
  preserving not beat-counting for measures; …). Each becomes a
  paragraph.
* **Show failure cases.** A thesis with no failure analysis is less
  credible than one that names its limitations. Include a "where this
  doesn't work" sub-section for each stage with one or two qualitative
  figures.
* **Reproducibility is a section, not an afterthought.** Master's
  examiners check this directly.

---

*Generated as a methodology scaffold for the OMR pipeline at*
`S:\omr` *— last regenerated automatically; refine prose in place.*
