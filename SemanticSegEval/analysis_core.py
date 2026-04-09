# analysis_core.py  (rasterio-FREE — uses osgeo.gdal, always bundled with QGIS)
# Core analysis functions for Semantic Segmentation Evaluation Plugin
# Based on notebook by Marek Bundzel, TU Kosice
#
# Dependencies: osgeo.gdal (bundled with QGIS), numpy (bundled), cv2 (opencv-python)
#
# Bug fixes vs. original notebook:
#   1. calculateClassifErrors: *2.0 produced float64 Errors. Fixed: int16 arithmetic.
#   2. extendToRadius: diamond kernel underestimated dilation by ~41% at radius=10.
#      Fixed: full 3x3 all-ones (square/Chebyshev) kernel.
#   3. analyzeAccuracyPerObject: loop indexed 0..N-1 but cv2 labels start at 1.
#      Fixed: iterate actual label IDs 1..num_labels-1 via a hit dict.
#   4. analyzeAccuracyPerObject: added explicit uint8 cast before cv2 calls.

import numpy as np
import cv2
import os


# ---------------------------------------------------------------------------
# GeoTIFF I/O  (osgeo.gdal — bundled with QGIS, no extra install needed)
# ---------------------------------------------------------------------------

def read_single_band_geotiff(path):
    """Open a single-band GeoTIFF; return (numpy_array, open_gdal_dataset)."""
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise IOError(f"GDAL could not open: {path}")
    if ds.RasterCount != 1:
        raise ValueError(
            f"Raster must have exactly 1 band (found {ds.RasterCount}): {path}")
    arr = ds.GetRasterBand(1).ReadAsArray()
    return arr, ds


def saveErrorsAsGeotiff(Errors, reference_tif_path, output_path):
    """
    Save the int16 Errors array as a compressed GeoTIFF, copying
    the geotransform and projection from reference_tif_path.
    """
    from osgeo import gdal
    gdal.UseExceptions()

    ref_ds = gdal.Open(reference_tif_path, gdal.GA_ReadOnly)
    if ref_ds is None:
        raise IOError(f"Cannot open reference raster: {reference_tif_path}")
    geotransform = ref_ds.GetGeoTransform()
    projection   = ref_ds.GetProjection()
    ref_ds = None

    rows, cols = Errors.shape
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path, cols, rows, 1, gdal.GDT_Int16,
        options=["COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=9",
                 "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"]
    )
    if out_ds is None:
        raise IOError(f"GDAL could not create: {output_path}")

    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)

    band = out_ds.GetRasterBand(1)
    band.WriteArray(Errors.astype(np.int16))
    band.SetMetadataItem("VALUE_NEG1", "False Negative (-1)")
    band.SetMetadataItem("VALUE_0",    "True Negative (0)")
    band.SetMetadataItem("VALUE_1",    "False Positive (1)")
    band.SetMetadataItem("VALUE_2",    "True Positive (2)")
    out_ds.FlushCache()
    out_ds = None


def make_error_path(tif_path):
    """Return sibling path with 'Errors_' prepended to the filename."""
    directory, filename = os.path.split(tif_path)
    name, ext = os.path.splitext(filename)
    return os.path.join(directory, f"Errors_{name}{ext}")


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def extendToRadius(binary_img, radius):
    """
    Dilate binary_img with a square (Chebyshev) structuring element.
    Bug fix: original used a diamond kernel (corners zeroed), reducing the
    dilation area by ~41% at radius=10 and deflating MOR10R.
    """
    img = binary_img.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)          # square structuring element
    extended = cv2.dilate(img, kernel, iterations=radius)
    return extended == 1


def intersectionOverUnion(prediction, groundTruth):
    pr  = prediction  > 0;  gt  = groundTruth > 0
    iou_fore = np.sum(np.logical_and(pr, gt))  / np.sum(np.logical_or(pr, gt))
    pb  = prediction  < 1;  gb  = groundTruth < 1
    iou_bgr  = np.sum(np.logical_and(pb, gb))  / np.sum(np.logical_or(pb, gb))
    return iou_fore, iou_bgr, (iou_fore + iou_bgr) / 2


def countMisclassifiedOutOfRadius(errors, radius):
    TP_mask       = errors == 2
    Misclassified = np.logical_or(errors == -1, errors == 1)
    TP_expanded   = extendToRadius(TP_mask, radius)
    return int(np.sum(Misclassified)) - int(np.sum(np.logical_and(Misclassified, TP_expanded)))


def calculateClassifErrors(Prediction, GT):
    """
    Returns int16 array:  0=TN, 2=TP, -1=FN, 1=FP.
    Bug fix: original *2.0 produced float64. Fixed: int16 throughout.
    """
    P = Prediction.astype(np.int16)
    G = GT.astype(np.int16)
    Errors = ((P + G) > 1).astype(np.int16) * 2
    Errors = Errors + (P - G)
    return Errors.astype(np.int16)


def calculateStatistics(Errors):
    predicted = (Errors == 2).astype(np.uint8) + (Errors == 1).astype(np.uint8)
    GT_rec    = (Errors == 2).astype(np.uint8) + (Errors == -1).astype(np.uint8)

    TN = int(np.sum(Errors == 0))
    TP = int(np.sum(Errors == 2))
    FN = int(np.sum(Errors == -1))
    FP = int(np.sum(Errors == 1))
    P  = TP + FN;  N = TN + FP

    _safe = lambda a, b: a / b if b > 0 else float('nan')
    TPR = _safe(TP, TP + FN)
    TNR = _safe(TN, TN + FP)
    PPV = _safe(TP, TP + FP)
    NPV = _safe(TN, TN + FN)
    F1  = _safe(2 * TP, 2 * TP + FP + FN)
    BalancedAccuracy = (TPR + TNR) / 2
    Accuracy         = _safe(TP + TN, P + N)
    MOR10R           = _safe(countMisclassifiedOutOfRadius(Errors, 10), P + N)
    IoU_fore, IoU_bgr, IoU_ave = intersectionOverUnion(predicted, GT_rec)
    denom = np.sqrt(float((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    MCC   = (TP * TN - FP * FN) / denom if denom > 0 else float('nan')

    return dict(
        BalancedAccuracy=BalancedAccuracy, Accuracy=Accuracy,
        TN=float(TN), TP=float(TP), FN=float(FN), FP=float(FP),
        TPR=TPR, TNR=TNR, PPV=PPV, NPV=NPV, F1=F1, MOR10R=MOR10R,
        IoU_fore=IoU_fore, IoU_bgr=IoU_bgr, IoU_ave=IoU_ave, MCC=MCC,
    )


# ---------------------------------------------------------------------------
# Per-object analysis
# ---------------------------------------------------------------------------

def analyzeAccuracyPerObject(GT, Prediction, connectivity=8, NoOfBins=400,
                              plot_title="Histogram of Total and Detected Objects Counts",
                              xlabel="Object Size (pixels)",
                              ylabel="No. of Objects",
                              legend_all="All Objects",
                              legend_detected="Detected Objects",
                              color_all="#1f77b4",
                              color_detected="#ff7f0e",
                              x_range=None,
                              _return_figure_only=False):
    """
    Per-object detection accuracy analysis + histogram figure.
    Returns (results_dict, matplotlib_figure, list_of_message_strings).

    _return_figure_only=True creates the Figure without pyplot.subplots()
    so no Qt window is opened from a worker thread (used by the QGIS plugin).
    """
    from matplotlib.figure import Figure as MplFigure

    messages = []

    # Bug fix 4: explicit uint8 cast required by cv2
    GT_u8   = GT.astype(np.uint8)
    Pred_u8 = Prediction.astype(np.uint8)

    # ── GT connected components ─────────────────────────────────────────────
    num_labels, labeledGT, stats, _ = cv2.connectedComponentsWithStats(
        GT_u8, cv2.CC_STAT_AREA, connectivity=connectivity)

    no_of_GT_objects = num_labels - 1
    messages.append(f"Number of GT objects: {no_of_GT_objects}")

    sizesGT     = stats[1:, 4]
    ave_size_GT = float(np.sum(sizesGT) / len(sizesGT)) if len(sizesGT) > 0 else 0.0
    min_size_GT = int(np.min(sizesGT))  if len(sizesGT) > 0 else 0
    max_size_GT = int(np.max(sizesGT))  if len(sizesGT) > 0 else 0
    messages.append(f"Average size of GT object: {ave_size_GT:.5f}")
    messages.append(f"Minimal size of GT object: {min_size_GT}")
    messages.append(f"Maximal size of GT object: {max_size_GT}")

    hit_img     = labeledGT * Pred_u8
    hits_u      = np.unique(hit_img, return_counts=True)
    hits        = hits_u[0][1:]
    hit_counts  = hits_u[1][1:]
    no_of_GT_objects_detected = len(hits)
    messages.append(f"Number of GT objects that were detected: {no_of_GT_objects_detected}")
    ave_hit_px = float(np.sum(hit_counts) / len(hit_counts)) if len(hit_counts) > 0 else 0.0
    messages.append(f"Average number of hit pixels per detected GT object: {ave_hit_px:.5f}")

    # Bug fix 3: iterate actual label IDs (1…num_labels-1)
    hit_dict       = {int(hits[i]): int(hit_counts[i]) for i in range(len(hits))}
    hited_sizes    = []
    no_hited_sizes = []
    for label_id in range(1, num_labels):
        sz = int(stats[label_id, 4])
        (hited_sizes if label_id in hit_dict else no_hited_sizes).append(sz)

    # ── Prediction connected components ─────────────────────────────────────
    num_labelsPred, labeledPred, statsPred, _ = cv2.connectedComponentsWithStats(
        Pred_u8, cv2.CC_STAT_AREA, connectivity=connectivity)

    no_of_Pred_objects = num_labelsPred - 1
    messages.append(f"Number of Predicted objects: {no_of_Pred_objects}")

    sizesPred     = statsPred[1:, 4]
    ave_size_Pred = float(np.sum(sizesPred) / len(sizesPred)) if len(sizesPred) > 0 else 0.0
    min_size_Pred = int(np.min(sizesPred)) if len(sizesPred) > 0 else 0
    max_size_Pred = int(np.max(sizesPred)) if len(sizesPred) > 0 else 0
    messages.append(f"Average size of Predicted object: {ave_size_Pred:.5f}")
    messages.append(f"Minimal size of Predicted object: {min_size_Pred}")
    messages.append(f"Maximal size of Predicted object: {max_size_Pred}")

    hit_img_pred = labeledPred * GT_u8
    hits_pred_u  = np.unique(hit_img_pred, return_counts=True)
    no_of_Pred_TP = len(hits_pred_u[0][1:])
    no_of_Pred_FP = no_of_Pred_objects - no_of_Pred_TP
    messages.append(f"Number of True Positive Prediction objects: {no_of_Pred_TP}")
    messages.append(f"Number of False Positive Prediction objects: {no_of_Pred_FP}")

    # ── Histogram ───────────────────────────────────────────────────────────
    sizes = np.sort(sizesGT)
    x_min = float(x_range[0]) if x_range else float(np.min(sizes))
    x_max = float(x_range[1]) if x_range else float(np.max(sizes)) + 1
    bns   = np.linspace(x_min, x_max, num=NoOfBins)

    # Always use the OO Figure API so no pyplot window is opened from a
    # worker thread (which crashes Qt on Windows).
    fig = MplFigure(figsize=(16, 8))
    ax  = fig.add_subplot(111)
    ax.hist(sizes,       bins=bns, label=legend_all,      color=color_all)
    ax.hist(hited_sizes, bins=bns, label=legend_detected, color=color_detected)
    ax.legend(prop={'size': 10})
    ax.set_ylabel(ylabel);  ax.set_xlabel(xlabel);  ax.set_title(plot_title)
    ax.set_xlim(x_min, x_max)
    # Use explicit bottom margin so the X axis label is never clipped on save.
    fig.subplots_adjust(bottom=0.12, top=0.93, left=0.07, right=0.98)

    return dict(
        no_of_GT_objects=no_of_GT_objects,
        ave_size_GT=ave_size_GT, min_size_GT=min_size_GT, max_size_GT=max_size_GT,
        no_of_GT_objects_detected=no_of_GT_objects_detected, ave_hit_px=ave_hit_px,
        no_of_Pred_objects=no_of_Pred_objects,
        ave_size_Pred=ave_size_Pred, min_size_Pred=min_size_Pred, max_size_Pred=max_size_Pred,
        no_of_Pred_TP=no_of_Pred_TP, no_of_Pred_FP=no_of_Pred_FP,
    ), fig, messages
