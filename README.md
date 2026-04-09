# Semantic Segmentation Evaluator — QGIS Plugin

A QGIS 3 plugin for evaluating the quality of binary semantic segmentation masks against a Ground Truth, designed primarily for archaeological LiDAR data but applicable to any binary segmentation task.

**Author:** Marek Bundzel, TU Košice — [marek.bundzel@tuke.sk](mailto:marek.bundzel@tuke.sk)

---

## Features

- Select Ground Truth and Prediction rasters directly from the layers loaded in your QGIS project
- Computes the **error map** (TN / TP / FN / FP) and saves it as a GeoTIFF, automatically added to the project above the Prediction layer with colour-coded symbology
- Reports **pixel-level quality statistics**: Accuracy, Balanced Accuracy, TPR, TNR, PPV, NPV, F1, MCC, IoU (foreground / background / average), and MOR10R
- **Per-object analysis** via connected components: detection rate, size statistics, TP/FP object counts
- **Embedded histogram** — resizable within QGIS, exportable as PNG (≥ 600 DPI) or vector SVG / PDF
- Progress bar with elapsed time; results copyable to clipboard for direct use in papers

Quality measures are explained in section 4.5 of:
> Bundzel et al. (2020). *Semantic Segmentation-Based Automatic Extraction of Archaeological Sites from Airborne Laser Scanning Data.* Remote Sensing, 12(22), 3685. https://doi.org/10.3390/rs12223685

---

## Requirements

| Component | Requirement |
|---|---|
| QGIS | 3.16 or newer (LTR recommended) |
| OS | Windows 10/11, macOS 12+, Linux (Ubuntu 20.04+) |
| Python | 3.9+ (bundled with QGIS) |
| RAM | 4 GB minimum; 16 GB recommended for large rasters |

### Python dependencies

| Library | Status |
|---|---|
| `osgeo.gdal` | Bundled with QGIS — no action needed |
| `numpy` | Bundled with QGIS — no action needed |
| `matplotlib` | Bundled with QGIS — no action needed |
| `opencv-python` | **Must be installed once** (see below) |

---

## Installation

### Step 1 — Install opencv-python into QGIS's Python

**Windows** — open the **OSGeo4W Shell** (not a regular terminal) and run:
```
python -m pip install opencv-python
```

**macOS** — open Terminal and run:
```
/Applications/QGIS.app/Contents/MacOS/bin/python3 -m pip install opencv-python
```

**Linux** — open a terminal and run:
```
python3 -m pip install opencv-python --break-system-packages
```

Verify the installation in the QGIS Python Console:
```python
import cv2
print(cv2.__version__)
```

### Step 2 — Install the plugin from ZIP

1. Download `SemanticSegEval.zip` from the [Releases](../../releases/latest) page
2. In QGIS: **Plugins → Manage and Install Plugins → Install from ZIP**
3. Browse to the downloaded ZIP and click **Install Plugin**
4. Go to the **Installed** tab and make sure **Semantic Segmentation Evaluator** is ticked

The plugin appears under **Raster → Semantic Seg Eval**.

---

## Usage

### Inputs & Run tab

- Select your **Ground Truth** and **Prediction** raster layers from the dropdowns (only rasters currently loaded in the project are shown)
- Both rasters must have exactly **one band** containing only values **0** (background) and **1** (object), and must have the **same shape**
- Set **Connectivity** (4 or 8) and **Number of histogram bins**, then click **▶ Run Analysis**

### Plot Settings tab

Customise the histogram title, axis labels, legend text, bar colours, and X axis range before or after running. Save the histogram via the **Save Histogram** button — use `.svg` or `.pdf` for vector output.

### Results tab

Displays all pixel-level statistics and per-object analysis results. Click **Copy to Clipboard** to paste into a spreadsheet or paper.

### Histogram tab

Shows the embedded, resizable histogram. Use the matplotlib toolbar (zoom, pan, save) directly on the canvas.

### Output error map

Saved alongside the Prediction file as `Errors_<prediction_filename>.tif` and added to the project with this symbology:

| Value | Class | Colour |
|---|---|---|
| 0 | TN — True Negative | Black, fully transparent |
| 2 | TP — True Positive | White, 70 % opacity |
| -1 | FN — False Negative | Blue, 70 % opacity |
| 1 | FP — False Positive | Red, 70 % opacity |

---

## Input data format

Both the Ground Truth and Prediction rasters must be:
- **Single-band GeoTIFF**
- Containing only pixel values **0** and **1**
- Identical **extent, resolution, and CRS**

To binarise a probability map (values 0–255 or 0.0–1.0) use the QGIS Raster Calculator before running the plugin.

---

## Citing this work

If you use this plugin in a scientific publication, please cite:

```
Bundzel, M.; Jaščur, M.; Kováč, M.; Lieskovský, T.; Malíček, T.; Šimkovič, M.
Semantic Segmentation-Based Automatic Extraction of Archaeological Sites from
Airborne Laser Scanning Data.
Remote Sensing 2020, 12, 3685.
https://doi.org/10.3390/rs12223685
```

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Changelog

### v1.0.0
- Initial release
