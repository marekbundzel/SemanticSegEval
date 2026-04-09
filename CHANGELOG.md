# Changelog

All notable changes to the Semantic Segmentation Evaluator QGIS plugin are documented here.

---

## [1.0.0] — 2025

### Added
- Select Ground Truth and Prediction rasters from project layer dropdowns
- Pixel-level statistics: Accuracy, Balanced Accuracy, TPR, TNR, PPV, NPV, F1, MCC, IoU (foreground / background / average), MOR10R
- Per-object connected-component analysis with size histogram
- Histogram embedded inside QGIS dialog (no separate matplotlib window, no crash on resize)
- Progress bar with elapsed time on the Inputs tab
- Error map GeoTIFF saved alongside the Prediction file, inserted above the Prediction layer in the project
- Colour-coded symbology applied automatically: TN black/transparent, TP white 70 %, FN blue 70 %, FP red 70 %
- Collision-safe error path: appends _1, _2, … if a same-named layer is already loaded
- Results text copyable to clipboard
- Histogram exportable as PNG (configurable DPI, default 600) or vector SVG / PDF
- All raster I/O via osgeo.gdal (bundled with QGIS) — no rasterio dependency

### Bug fixes (vs. original notebook)
- `calculateClassifErrors`: replaced `*2.0` with integer arithmetic to avoid float64 Errors array
- `extendToRadius`: replaced diamond kernel with full 3×3 all-ones kernel (square dilation), correcting ~41 % underestimation of MOR10R
- `analyzeAccuracyPerObject`: fixed off-by-one label index (cv2 labels start at 1, not 0)
- `analyzeAccuracyPerObject`: added explicit uint8 cast before cv2 connected-components calls
