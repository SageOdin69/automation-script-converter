# How to Run — TestScriptGeneratorLibrary

## What It Does
Reads a `.txt` test script (PDL + UP/UM statements), crops UI elements
from your reference screenshots, and writes a ready-to-run pytest `.py` file.

---

## Project Folder Structure (recommended)

```
your_project/
│
├── libraries/
│   ├── TestScriptGeneratorLibrary.py   ← the generator library
│   └── CyfastFapLibrary.py             ← existing automation library
│
├── scripts/                            ← your source .txt test scripts
│   └── tc_fap_04_5005.txt
│
├── ref_screenshots/                    ← full reference HSI screenshots
│   ├── 1.png   (master page)
│   ├── 2.png   (flyout expanded)
│   ├── 3.png   (cabin lighting main)
│   └── 4.png   (entire scenario dialog)
│
├── ref/                                ← AUTO-GENERATED cropped images go here
│   └── CIL_/
│       ├── Select_Button.png
│       ├── Breakfast_scenario.png
│       ├── cancel.png
│       └── ...
│
└── generated_tests/                    ← AUTO-GENERATED pytest scripts go here
    └── test_tc_fap_04_5005.py
```

---

## Step 1 — Install Dependencies

Run this once on your machine:

```bash
pip install rapidocr-onnxruntime opencv-python pillow numpy
```

If you want VLM-assisted cropping (Ollama, optional — better accuracy):
```bash
pip install ollama
ollama pull qwen2-vl        # or: ollama pull llama3.2-vision
```

---

## Step 2 — Run via Python Script (recommended)

Create a file called `run_generator.py` in your project root:

```python
import sys
sys.path.insert(0, "./libraries")          # so Python finds your libraries folder

from TestScriptGeneratorLibrary import TestScriptGeneratorLibrary

gen = TestScriptGeneratorLibrary()         # uses OCR only (no VLM needed)

gen.generate(
    # ── Source script ────────────────────────────────────────
    script_path = "scripts/tc_fap_04_5005.txt",

    # ── Reference screenshots — one per screen state ─────────
    # Format: (image_path, label)
    reference_images = [
        ("ref_screenshots/1.png", "master_page"),
        ("ref_screenshots/2.png", "flyout_expanded"),
        ("ref_screenshots/3.png", "cabin_lighting_main"),
        ("ref_screenshots/4.png", "entire_scenario_dialog"),
    ],

    # ── Where to save the generated pytest file ───────────────
    output_path = "generated_tests/test_tc_fap_04_5005.py",

    # ── Where to save the cropped reference images ────────────
    ref_output_dir = "./ref/CIL_/",

    # ── Which reference image belongs to which UP statement ───
    # UP index (0-based) : image path
    up_image_map = {
        0: "ref_screenshots/3.png",   # "Verify Select Button visible"
        1: "ref_screenshots/4.png",   # "Verify 12 scenarios panel"
        2: "ref_screenshots/4.png",   # "Verify Breakfast indicated as Run"
        3: "ref_screenshots/4.png",   # "Click cancel, verify panel invisible"
        4: "ref_screenshots/3.png",   # "Proceed to finish" — no crop needed
        5: "ref_screenshots/3.png",   # "Script finished"  — no crop needed
    },

    # ── Which reference image belongs to which UM statement ───
    um_image_map = {
        0: "ref_screenshots/2.png",   # "Open fly-out menu..." -> flyout image
    },
)
```

Then run it:

```bash
cd your_project
python run_generator.py
```

---

## Step 3 — What Gets Created

After running, you will find:

### Cropped Reference Images  →  `./ref/CIL_/`
| File | What it is | Source image |
|---|---|---|
| `Select_Button.png` | The "Select" button in Entire Scenario section | Image 3 |
| `Breakfast_scenario.png` | The "Entire 1.8 / Run" cell | Image 4 |
| `cancel.png` | The Cancel button at the bottom of the dialog | Image 4 |
| `scenario_panel_is.png` | The Entire Scenario label (used to verify it's gone) | Image 4 |
| `clicking_select_on_Entire_Scen.png` | Scenario panel header region | Image 4 |

### Generated Pytest File  →  `./generated_tests/test_tc_fap_04_5005.py`
Ready to run with your existing conftest.py and `cyfastobj` fixture.

---

## Step 4 — Run via Command Line (alternative)

You can also run directly from terminal without writing a Python script:

```bash
# Basic usage
python TestScriptGeneratorLibrary.py \
    scripts/tc_fap_04_5005.txt \
    ref_screenshots/1.png:master_page \
    ref_screenshots/2.png:flyout_expanded \
    ref_screenshots/3.png:cabin_lighting_main \
    ref_screenshots/4.png:entire_scenario_dialog \
    --ref-dir ./ref/CIL_/ \
    --output generated_tests/test_tc_fap_04_5005.py

# Just see the parse + intent summary (no files written)
python TestScriptGeneratorLibrary.py scripts/tc_fap_04_5005.txt --summary

# With VLM (Ollama must be running)
python TestScriptGeneratorLibrary.py scripts/tc_fap_04_5005.txt \
    ref_screenshots/3.png:main ref_screenshots/4.png:dialog \
    --vlm --vlm-model qwen2-vl
```

---

## Step 5 — Run the Generated Pytest

```bash
pytest generated_tests/test_tc_fap_04_5005.py -v
```

---

## How `up_image_map` Works

This is the most important parameter. It tells the library **which screenshot** 
to look at when cropping the element for each UP statement.

```
UP[0] = "Verify that a Select Button is visible..."
         → You must tell it: look in Image 3 (cabin lighting page)
           because that's the screen where the Select button appears.

UP[2] = "Verify that Breakfast scenario is indicated as Run."
         → You must tell it: look in Image 4 (scenario dialog)
           because "Run" text only appears after the dialog opens.
```

**Without `up_image_map`**, the library does auto-assignment by scoring which 
image's OCR text best matches each UP statement. This works for most cases but 
may get it wrong when the UP text doesn't share words with the visible screen.
Always provide the map when you know the order of screens.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: rapidocr_onnxruntime` | Run `pip install rapidocr-onnxruntime` |
| `FileNotFoundError: Cannot read image` | Check image paths are correct relative to where you run Python |
| Crop is the full image (fallback) | The keyword wasn't found via OCR — either add it to `up_image_map` or use VLM mode |
| Wrong element cropped | Increase `crop_padding` or use `up_image_map` to point to the correct image |
| Generated test has `"None"` as crop path | UP was classified as PROCEED/UNKNOWN — check the raw UP text |
