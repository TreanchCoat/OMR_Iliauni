# OMR Pipeline — Project Layout

End-to-end Optical Music Recognition pipeline: PNG score → MusicXML.

```
S:\omr\
├── .env                 ← local config (gitignored)
├── .env.example         ← documented template
├── requirements.txt
├── README.md            ← this file
│
├── src\                 ← pipeline source modules
│   ├── env_loader.py        — .env loader + path bootstrap
│   ├── xml_builder.py       — lxml-based MusicXML constructor
│   ├── score_analyzer.py    — staff detection + row clustering
│   ├── staff_remover.py     — staff-line removal
│   ├── staff_unet.py        — U-Net architecture
│   ├── staff_detector_unet.py
│   ├── staff_detector_yolo.py
│   ├── staff_rectifier.py
│   ├── preprocessing.py     — Stage 2 orchestrator
│   ├── symbol_detector.py   — Stage 3 (YOLO inference)
│   ├── bbox_refiner.py      — Stage 3 polish
│   ├── build_musicxml_v2.py — Stage 4 (chords/rests/fermatas/...)
│   ├── measure_recalculator.py — Stage 4½ (re-bar by <time>)
│   └── score_to_xml.py      — legacy v1 (kept for reference)
│
├── scripts\             ← entry-point CLIs
│   ├── pipeline.py             — full image → MusicXML
│   └── preprocess_for_training.py — batch staff-crop generator
│
├── api\                 ← REST API + clients
│   ├── real_api.py             — runs the real pipeline
│   ├── dummy_api.py            — serves sample fixtures
│   ├── api_client_example.py
│   ├── process_score.py
│   └── api_tester.html         — browser tester UI
│
├── training\            ← model-training scripts
│   ├── staff_unet_train.py
│   ├── dataset_prep.py
│   └── muscima_manifest.json
│
├── diagnostics\         ← ad-hoc dev/debug scripts
│   ├── diagnose_loaded.py
│   ├── diagnose_mask.py
│   ├── diagnose_unet_output.py
│   └── visualizer.py
│
├── tests\               ← smoke / regression tests
│   └── _smoke_test_v2.py
│
├── models\              ← trained weights (Git LFS)
│   ├── staff_unet.pth
│   ├── ola_v2.pt
│   └── deepscores_crops_v1.pt
│
├── data\
│   ├── input\               — input score PNGs
│   └── sample\              — dummy-API fixtures
│
├── docs\
│   ├── thesis_methodology.md
│   ├── UNET_SETUP.md
│   └── Run command.txt
│
└── archive\             ← legacy code + old test outputs
```

## Quick start

```cmd
:: One-off OMR
python scripts\pipeline.py data\input\page1.png output\

:: Generate training crops from a folder of pages
python scripts\preprocess_for_training.py data\input\ output\staves\

:: Re-bar an XML after manual edits
python src\measure_recalculator.py score.xml score_rebar.xml --time 3/4

:: Dev API
python api\real_api.py
```

## Configuration

All paths and runtime settings are read from `.env` (loaded automatically
by `src\env_loader.py`).  The supported variables are documented in
`.env.example`.  Defaults derived from the project root keep everything
working with zero `.env` setup; the file is needed only when you want
data outside the repo.

The `env_loader` exposes path constants you can import in any script:

```python
import env_loader
print(env_loader.MODEL_PATH)        # YOLO weights
print(env_loader.INPUT_DIR)         # default input folder
print(env_loader.OUTPUT_BASE_DIR)   # default output folder
```

## Pipeline stages

```
PNG page ──► [1] staff_rectifier     ──► rectified.png
             [2] preprocessing       ──► ProcessedScore
             [3] symbol_detector     ──► PageDetections
             [4] build_musicxml_v2   ──► score.xml
             [4½] measure_recalculator → score.xml (re-barred)
```

See `docs\thesis_methodology.md` for a stage-by-stage methodology
write-up.
