# dialog_main.py
"""
Semantic Segmentation Evaluation Plugin – Main Dialog
Author: built around analysis_core.py (Marek Bundzel, TU Kosice)

Changes in this version:
  - Layer dropdowns replace file-browse fields
  - Progress bar + elapsed-time label on Inputs tab
  - Histogram embedded in plugin dialog (Qt canvas) → no separate matplotlib
    window, eliminates the Qt access-violation crash on resize/maximise
  - Error-path collision: if Errors_*.tif already exists AND is loaded in the
    project, an indexed suffix (_1, _2, …) is appended until the path is free
"""

import os
import time
import traceback

import numpy as np

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
    QGroupBox, QTextEdit, QFileDialog, QSizePolicy,
    QTabWidget, QWidget, QColorDialog, QDoubleSpinBox,
    QMessageBox, QApplication, QProgressBar, QSplitter,
    QScrollArea,
)
from qgis.PyQt.QtGui import QColor, QPalette
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal, QTimer

from qgis.core import (
    QgsProject, QgsRasterLayer,
    QgsPalettedRasterRenderer,
    QgsMapLayerType,
)

# Matplotlib embedded in Qt — use the Qt5Agg backend that shares Qt's event loop
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unique_error_path(base_path):
    """
    Return base_path if it does not collide with a layer already loaded in the
    current QGIS project.  If it does, append _1, _2, … until collision-free.
    Also avoid overwriting a file that GDAL still has locked (exists on disk AND
    is in the project).
    """
    project_sources = set()
    for layer in QgsProject.instance().mapLayers().values():
        src = layer.source()
        if src:
            project_sources.add(os.path.normpath(src))

    candidate = base_path
    directory, filename = os.path.split(base_path)
    name, ext = os.path.splitext(filename)
    idx = 1
    while (os.path.exists(candidate) and
           os.path.normpath(candidate) in project_sources):
        candidate = os.path.join(directory, f"{name}_{idx}{ext}")
        idx += 1
    return candidate


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class AnalysisWorker(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)        # text log
    step     = pyqtSignal(int)        # progress bar value 0-100
    error    = pyqtSignal(str)

    # Named steps with approximate % weight
    STEPS = [
        ("Loading Ground Truth raster…",          5),
        ("Loading Prediction raster…",            10),
        ("Calculating classification errors…",    20),
        ("Saving errors raster…",                 35),
        ("Calculating pixel-level statistics…",   60),
        ("Analysing per-object accuracy…",        80),
        ("Building histogram…",                   95),
    ]

    def __init__(self, params):
        super().__init__()
        self.params = params

    def _emit(self, idx):
        label, pct = self.STEPS[idx]
        self.progress.emit(label)
        self.step.emit(pct)

    def run(self):
        try:
            from .analysis_core import (
                calculateClassifErrors, saveErrorsAsGeotiff, make_error_path,
                calculateStatistics, analyzeAccuracyPerObject,
                read_single_band_geotiff,
            )
            import matplotlib
            matplotlib.use('Qt5Agg')
            from matplotlib.figure import Figure

            p = self.params

            # ── load GT ────────────────────────────────────────────────────
            self._emit(0)
            GT, _ds = read_single_band_geotiff(p['gt_path'])
            _ds = None
            unique_gt = set(np.unique(GT).tolist())
            if not unique_gt.issubset({0, 1}):
                raise ValueError(
                    f"GT raster must contain only 0 and 1 (found: {unique_gt}).")

            # ── load Prediction ────────────────────────────────────────────
            self._emit(1)
            Prediction, _ds = read_single_band_geotiff(p['pred_path'])
            _ds = None
            unique_pred = set(np.unique(Prediction).tolist())
            if not unique_pred.issubset({0, 1}):
                raise ValueError(
                    f"Prediction raster must contain only 0 and 1 (found: {unique_pred}).")
            if GT.shape != Prediction.shape:
                raise ValueError(
                    f"Shape mismatch: GT {GT.shape} vs Prediction {Prediction.shape}.")

            # ── errors array ───────────────────────────────────────────────
            self._emit(2)
            Errors = calculateClassifErrors(Prediction, GT)

            # ── save raster (collision-safe path resolved in main thread,
            #    passed in via params) ────────────────────────────────────
            self._emit(3)
            error_path = p['error_path']
            self.progress.emit(f"  → {error_path}")
            saveErrorsAsGeotiff(Errors, p['pred_path'], error_path)

            # ── pixel statistics ───────────────────────────────────────────
            self._emit(4)
            Statistics = calculateStatistics(Errors)

            # ── per-object ─────────────────────────────────────────────────
            self._emit(5)
            # analyzeAccuracyPerObject creates a Figure internally;
            # we use a non-interactive Figure so no Qt window is opened.
            per_obj_results, fig, obj_messages = analyzeAccuracyPerObject(
                GT, Prediction,
                connectivity=p['connectivity'],
                NoOfBins=p['no_of_bins'],
                plot_title=p['plot_title'],
                xlabel=p['xlabel'],
                ylabel=p['ylabel'],
                legend_all=p['legend_all'],
                legend_detected=p['legend_detected'],
                color_all=p['color_all'],
                color_detected=p['color_detected'],
                x_range=p.get('x_range'),
                # Request a plain Figure (no pyplot show) from analysis_core
                _return_figure_only=True,
            )

            self._emit(6)
            self.step.emit(100)

            self.finished.emit(dict(
                error_path=error_path,
                Statistics=Statistics,
                per_obj_results=per_obj_results,
                obj_messages=obj_messages,
                fig=fig,
                pred_path=p['pred_path'],
                connectivity=p['connectivity'],
            ))

        except Exception:
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Colour-picker button
# ---------------------------------------------------------------------------

class ColorButton(QPushButton):
    def __init__(self, color='#1f77b4', parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._refresh()
        self.clicked.connect(self._pick)

    def _refresh(self):
        self.setText(self._color.name())
        pal = self.palette()
        pal.setColor(QPalette.Button, self._color)
        pal.setColor(QPalette.ButtonText,
                     QColor('black') if self._color.lightness() > 128 else QColor('white'))
        self.setAutoFillBackground(True)
        self.setPalette(pal)

    def _pick(self):
        c = QColorDialog.getColor(self._color, self)
        if c.isValid():
            self._color = c
            self._refresh()

    def color(self):
        return self._color.name()


# ---------------------------------------------------------------------------
# Embedded matplotlib canvas (placed on the Histogram tab)
# ---------------------------------------------------------------------------

class HistogramWidget(QWidget):
    """A resizable widget that hosts a matplotlib Figure inside Qt."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fig    = None
        self._canvas = None
        self._toolbar = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # placeholder label
        self._placeholder = QLabel("Run the analysis to see the histogram here.")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._placeholder)

    def set_figure(self, fig):
        """Replace the current figure with fig (a matplotlib Figure object)."""
        # Remove old widgets
        if self._toolbar:
            self._layout.removeWidget(self._toolbar)
            self._toolbar.deleteLater()
            self._toolbar = None
        if self._canvas:
            self._layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None
        if self._placeholder:
            self._layout.removeWidget(self._placeholder)
            self._placeholder.deleteLater()
            self._placeholder = None

        self._fig = fig

        # Attach the figure to a Qt canvas
        self._canvas = FigureCanvas(fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Navigation toolbar (zoom, pan, save) — safe inside Qt event loop
        self._toolbar = NavigationToolbar(self._canvas, self)

        self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas)

        # subplots_adjust bottom gives room for the X axis label
        fig.subplots_adjust(bottom=0.12)
        self._canvas.draw()

    def save_figure(self, path, dpi):
        if self._fig is None:
            return False
        # bbox_inches='tight' + pad_inches ensures axis labels are not clipped.
        # We also temporarily set the figure DPI to the requested value so that
        # the saved pixel dimensions match dpi * figure_size_in_inches.
        orig_dpi = self._fig.get_dpi()
        try:
            self._fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.3)
        finally:
            self._fig.set_dpi(orig_dpi)
        return True


# ---------------------------------------------------------------------------
# Main Dialog
# ---------------------------------------------------------------------------

class SemanticSegEvalDialog(QDialog):

    PAPER_URL  = "https://www.mdpi.com/2072-4292/12/22/3685"
    MANUAL_URL = ("https://github.com/marekbundzel/SemanticSegEval"
                  "/raw/main/SemanticSegEval_InstallationAndUserManual_v1.docx")

    STAT_LABELS = {
        'BalancedAccuracy': 'Balanced Accuracy',
        'Accuracy':         'Accuracy (Overall)',
        'TN':               'True Negatives (TN)',
        'TP':               'True Positives (TP)',
        'FN':               'False Negatives (FN)',
        'FP':               'False Positives (FP)',
        'TPR':              'True Positive Rate / Sensitivity (TPR)',
        'TNR':              'True Negative Rate / Specificity (TNR)',
        'PPV':              'Positive Predictive Value / Precision (PPV)',
        'NPV':              'Negative Predictive Value (NPV)',
        'F1':               'F1 Score',
        'MOR10R':           'Misclassified Outside Radius 10 Rate (MOR10R) [see paper §4.5]',
        'IoU_fore':         'IoU – Foreground',
        'IoU_bgr':          'IoU – Background',
        'IoU_ave':          'IoU – Average',
        'MCC':              'Matthews Correlation Coefficient (MCC)',
    }

    def __init__(self, iface, parent=None):
        super().__init__(parent or iface.mainWindow())
        self.iface      = iface
        self.worker     = None
        self._start_time = None
        self._timer     = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._update_elapsed)

        self.setWindowTitle("Semantic Segmentation Evaluator")
        self.setMinimumSize(820, 620)
        self.resize(900, 700)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._build_ui()

        # Populate dropdowns initially
        self._refresh_layer_lists()

        # Keep dropdowns fresh if project layers change
        QgsProject.instance().layersAdded.connect(self._refresh_layer_lists)
        QgsProject.instance().layersRemoved.connect(self._refresh_layer_lists)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = QVBoxLayout(self)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

        self._tabs.addTab(self._tab_inputs(),    "Inputs & Run")
        self._tabs.addTab(self._tab_plot(),      "Plot Settings")
        self._tabs.addTab(self._tab_results(),   "Results")
        self._tabs.addTab(self._tab_histogram(), "Histogram")

        # Run button
        self.btn_run = QPushButton("▶  Run Analysis")
        self.btn_run.setFixedHeight(36)
        self.btn_run.clicked.connect(self._run)
        root.addWidget(self.btn_run)

    # ---- Tab 1: Inputs ----

    def _tab_inputs(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        # --- Layer selection ---
        grp_layers = QGroupBox("Input Raster Layers (from current QGIS project)")
        g = QGridLayout(grp_layers)

        g.addWidget(QLabel("Ground Truth mask:"), 0, 0)
        self.cb_gt = QComboBox()
        self.cb_gt.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        g.addWidget(self.cb_gt, 0, 1)

        g.addWidget(QLabel("Prediction mask:"), 1, 0)
        self.cb_pred = QComboBox()
        self.cb_pred.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        g.addWidget(self.cb_pred, 1, 1)

        btn_refresh = QPushButton("↻  Refresh layer list")
        btn_refresh.clicked.connect(self._refresh_layer_lists)
        g.addWidget(btn_refresh, 2, 0, 1, 2)

        lay.addWidget(grp_layers)

        # --- Parameters ---
        grp_params = QGroupBox("Analysis Parameters")
        p = QGridLayout(grp_params)

        p.addWidget(QLabel("Connectivity:"), 0, 0)
        self.cb_conn = QComboBox()
        self.cb_conn.addItems(["4", "8"])
        self.cb_conn.setCurrentIndex(1)
        p.addWidget(self.cb_conn, 0, 1)

        p.addWidget(QLabel("Number of histogram bins:"), 1, 0)
        self.sp_bins = QSpinBox()
        self.sp_bins.setRange(10, 5000)
        self.sp_bins.setValue(400)
        p.addWidget(self.sp_bins, 1, 1)

        lay.addWidget(grp_params)

        # --- Progress ---
        grp_prog = QGroupBox("Progress")
        pv = QVBoxLayout(grp_prog)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        pv.addWidget(self.progress_bar)

        hrow = QHBoxLayout()
        self.lbl_step    = QLabel("Ready.")
        self.lbl_elapsed = QLabel("")
        hrow.addWidget(self.lbl_step, 1)
        hrow.addWidget(self.lbl_elapsed)
        pv.addLayout(hrow)

        lay.addWidget(grp_prog)

        # --- Help link ---
        lbl_manual = QLabel(
            '<a href="' + self.MANUAL_URL + '">'
            '📄 Installation & User Manual (download .docx)'
            '</a>'
        )
        lbl_manual.setOpenExternalLinks(True)
        lbl_manual.setToolTip(self.MANUAL_URL)
        lay.addWidget(lbl_manual)

        lay.addStretch()
        return w

    # ---- Tab 2: Plot Settings ----

    def _tab_plot(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        grp = QGroupBox("Histogram Appearance")
        g = QGridLayout(grp)
        row = 0

        def add_row(label, widget):
            nonlocal row
            g.addWidget(QLabel(label), row, 0)
            g.addWidget(widget, row, 1, 1, 3)
            row += 1

        self.le_title  = self._le("Histogram of Total and Detected Objects Counts")
        self.le_xlabel = self._le("Object Size (pixels)")
        self.le_ylabel = self._le("No. of Objects")
        self.le_leg_all = self._le("All Objects")
        self.le_leg_det = self._le("Detected Objects")

        add_row("Plot title:",            self.le_title)
        add_row("X axis label:",          self.le_xlabel)
        add_row("Y axis label:",          self.le_ylabel)
        add_row("Legend – All Objects:",  self.le_leg_all)
        add_row("Legend – Detected:",     self.le_leg_det)

        self.btn_col_all = ColorButton('#1f77b4')
        self.btn_col_det = ColorButton('#ff7f0e')
        add_row("Colour – All Objects:",  self.btn_col_all)
        add_row("Colour – Detected:",     self.btn_col_det)

        g.addWidget(QLabel("X axis range (0, 0 = auto):"), row, 0)
        hr = QHBoxLayout()
        self.sp_xmin = QDoubleSpinBox(); self.sp_xmin.setRange(0, 1e9); self.sp_xmin.setValue(0)
        self.sp_xmax = QDoubleSpinBox(); self.sp_xmax.setRange(0, 1e9); self.sp_xmax.setValue(0)
        hr.addWidget(self.sp_xmin); hr.addWidget(QLabel("to")); hr.addWidget(self.sp_xmax)
        g.addLayout(hr, row, 1, 1, 3); row += 1

        lay.addWidget(grp)

        # Save
        grp_save = QGroupBox("Save Histogram to File")
        sv = QGridLayout(grp_save)
        sv.addWidget(QLabel("Output file:"), 0, 0)
        self.le_hist_out = QLineEdit()
        sv.addWidget(self.le_hist_out, 0, 1)
        btn_hist_out = QPushButton("Browse…")
        btn_hist_out.clicked.connect(self._browse_hist_out)
        sv.addWidget(btn_hist_out, 0, 2)
        sv.addWidget(QLabel("DPI (raster formats):"), 1, 0)
        self.sp_dpi = QSpinBox(); self.sp_dpi.setRange(72, 2400); self.sp_dpi.setValue(600)
        sv.addWidget(self.sp_dpi, 1, 1)
        sv.addWidget(QLabel("Tip: .svg or .pdf for vector (DPI ignored)"), 2, 0, 1, 3)
        self.btn_save_hist = QPushButton("Save Histogram")
        self.btn_save_hist.setEnabled(False)
        self.btn_save_hist.clicked.connect(self._save_histogram)
        sv.addWidget(self.btn_save_hist, 3, 0, 1, 3)

        lay.addWidget(grp_save)
        lay.addStretch()
        return w

    # ---- Tab 3: Results ----

    def _tab_results(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        self.txt_results = QTextEdit()
        self.txt_results.setReadOnly(True)
        self.txt_results.setFontFamily("Monospace")
        lay.addWidget(self.txt_results)

        btn_copy = QPushButton("Copy to Clipboard")
        btn_copy.clicked.connect(
            lambda: QApplication.clipboard().setText(self.txt_results.toPlainText()))
        lay.addWidget(btn_copy)
        return w

    # ---- Tab 4: Histogram (embedded canvas) ----

    def _tab_histogram(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self._hist_widget = HistogramWidget()
        lay.addWidget(self._hist_widget)
        return w

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _le(default=""):
        w = QLineEdit(default)
        return w

    def _refresh_layer_lists(self, *args):
        """Repopulate both combo boxes with current raster layers."""
        gt_prev   = self.cb_gt.currentData()
        pred_prev = self.cb_pred.currentData()

        self.cb_gt.blockSignals(True)
        self.cb_pred.blockSignals(True)
        self.cb_gt.clear()
        self.cb_pred.clear()

        raster_layers = [
            layer for layer in QgsProject.instance().mapLayers().values()
            if layer.type() == QgsMapLayerType.RasterLayer
        ]

        if not raster_layers:
            self.cb_gt.addItem("— no raster layers in project —", None)
            self.cb_pred.addItem("— no raster layers in project —", None)
        else:
            for layer in raster_layers:
                self.cb_gt.addItem(layer.name(), layer.id())
                self.cb_pred.addItem(layer.name(), layer.id())

            # Restore previous selections if still present
            for cb, prev in [(self.cb_gt, gt_prev), (self.cb_pred, pred_prev)]:
                if prev:
                    idx = cb.findData(prev)
                    if idx >= 0:
                        cb.setCurrentIndex(idx)

        self.cb_gt.blockSignals(False)
        self.cb_pred.blockSignals(False)

    def _layer_source(self, combo):
        """Return the file path for the layer selected in combo, or None."""
        layer_id = combo.currentData()
        if not layer_id:
            return None
        layer = QgsProject.instance().mapLayer(layer_id)
        if layer and layer.isValid():
            return layer.source()
        return None

    # ------------------------------------------------------------------ Run

    def _run(self):
        gt_path   = self._layer_source(self.cb_gt)
        pred_path = self._layer_source(self.cb_pred)

        if not gt_path:
            QMessageBox.warning(self, "No layer selected",
                                "Please select a Ground Truth raster layer.")
            return
        if not pred_path:
            QMessageBox.warning(self, "No layer selected",
                                "Please select a Prediction raster layer.")
            return
        if gt_path == pred_path:
            QMessageBox.warning(self, "Same layer",
                                "Ground Truth and Prediction must be different layers.")
            return

        x_min   = self.sp_xmin.value()
        x_max   = self.sp_xmax.value()
        x_range = (x_min, x_max) if x_max > x_min else None

        # Resolve collision-safe output path HERE in the main thread
        # (needs access to QgsProject which is not thread-safe)
        from .analysis_core import make_error_path
        base_error_path = make_error_path(pred_path)
        error_path      = _unique_error_path(base_error_path)

        params = dict(
            gt_path         = gt_path,
            pred_path       = pred_path,
            error_path      = error_path,
            connectivity    = int(self.cb_conn.currentText()),
            no_of_bins      = self.sp_bins.value(),
            plot_title      = self.le_title.text(),
            xlabel          = self.le_xlabel.text(),
            ylabel          = self.le_ylabel.text(),
            legend_all      = self.le_leg_all.text(),
            legend_detected = self.le_leg_det.text(),
            color_all       = self.btn_col_all.color(),
            color_detected  = self.btn_col_det.color(),
            x_range         = x_range,
        )

        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳  Running…")
        self.txt_results.clear()
        self.progress_bar.setValue(0)
        self.lbl_step.setText("Starting…")
        self.lbl_elapsed.setText("")

        self._start_time = time.time()
        self._timer.start()

        self.worker = AnalysisWorker(params)
        self.worker.progress.connect(self._on_progress_text)
        self.worker.step.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    # ------------------------------------------------------------------ Slots

    def _update_elapsed(self):
        if self._start_time is not None:
            secs = time.time() - self._start_time
            self.lbl_elapsed.setText(f"{secs:.1f} s")

    def _on_progress_text(self, msg):
        self.lbl_step.setText(msg)
        self.txt_results.append(msg)

    def _on_finished(self, result):
        self._timer.stop()
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  Run Analysis")
        self.progress_bar.setValue(100)

        elapsed = time.time() - self._start_time if self._start_time else 0
        self.lbl_step.setText(f"✔ Done in {elapsed:.1f} s")
        self.lbl_elapsed.setText(f"{elapsed:.1f} s")

        # ── embed histogram ────────────────────────────────────────────────
        self._hist_widget.set_figure(result['fig'])
        self.btn_save_hist.setEnabled(True)
        # Switch to Histogram tab so user sees it immediately
        self._tabs.setCurrentIndex(3)

        # ── results text ───────────────────────────────────────────────────
        lines = []
        lines.append("=" * 60)
        lines.append("PIXEL-LEVEL STATISTICS")
        lines.append("=" * 60)
        lines.append("For explanation of quality measures (including MOR10R),")
        lines.append(f"see section 4.5 of: {self.PAPER_URL}")
        lines.append("")
        lines.append("Installation & User Manual:")
        lines.append(f"  {self.MANUAL_URL}")
        lines.append("")

        stats = result['Statistics']
        for key, label in self.STAT_LABELS.items():
            val = stats.get(key, float('nan'))
            try:
                if float(val) != int(float(val)):
                    lines.append(f"  {label}:\n      {float(val):.5f}")
                else:
                    lines.append(f"  {label}:\n      {int(float(val))}")
            except (ValueError, OverflowError):
                lines.append(f"  {label}:\n      {val}")

        lines.append("")
        lines.append("=" * 60)
        lines.append("PER-OBJECT ANALYSIS")
        lines.append("=" * 60)
        lines.append(f"  Connectivity: {result.get('connectivity', '?')}")

        for msg in result['obj_messages']:
            if ':' in msg:
                k, v = msg.split(':', 1)
                try:
                    fv = float(v.strip())
                    if fv != int(fv):
                        lines.append(f"  {k.strip()}:\n      {fv:.5f}")
                    else:
                        lines.append(f"  {k.strip()}:\n      {int(fv)}")
                except ValueError:
                    lines.append(f"  {msg}")
            else:
                lines.append(f"  {msg}")

        lines.append("")
        lines.append(f"Total elapsed time: {elapsed:.1f} s")

        self.txt_results.setPlainText('\n'.join(lines))

        # ── add errors layer to project ────────────────────────────────────
        self._add_errors_layer(result['error_path'], result['pred_path'])

    def _on_error(self, tb):
        self._timer.stop()
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  Run Analysis")
        self.progress_bar.setValue(0)
        self.lbl_step.setText("Error — see Results tab.")
        self.txt_results.append("\nERROR:\n" + tb)
        QMessageBox.critical(self, "Analysis Error", tb[:800])

    def _browse_hist_out(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Histogram As", "",
            "PNG (*.png);;SVG vector (*.svg);;PDF vector (*.pdf);;TIFF (*.tif *.tiff);;All files (*)")
        if path:
            self.le_hist_out.setText(path)

    def _save_histogram(self):
        path = self.le_hist_out.text().strip()
        if not path:
            QMessageBox.warning(self, "No output path",
                                "Please specify an output file in the Plot Settings tab.")
            return
        dpi = self.sp_dpi.value()
        try:
            ok = self._hist_widget.save_figure(path, dpi)
            if ok:
                QMessageBox.information(self, "Saved", f"Histogram saved to:\n{path}")
            else:
                QMessageBox.warning(self, "No figure", "Run the analysis first.")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    # ------------------------------------------------------------------ Layer

    def _add_errors_layer(self, error_path, pred_path):
        layer_name = os.path.basename(error_path)
        rl = QgsRasterLayer(error_path, layer_name)
        if not rl.isValid():
            self.txt_results.append(
                f"WARNING: could not load errors layer: {error_path}")
            return

        self._apply_symbology(rl)

        # addMapLayer(layer, False) registers it WITHOUT adding a tree node.
        # We then insert a tree node manually at the right position.
        QgsProject.instance().addMapLayer(rl, False)

        root = QgsProject.instance().layerTreeRoot()

        # Find the tree node for the Prediction layer
        pred_nodes = [
            n for n in root.findLayers()
            if n.layer() and
               os.path.normpath(n.layer().source()) == os.path.normpath(pred_path)
        ]

        if pred_nodes:
            pred_node = pred_nodes[0]
            parent    = pred_node.parent()
            idx       = parent.children().index(pred_node)
            # Insert the new layer node directly above the prediction node
            from qgis.core import QgsLayerTreeLayer
            new_node = QgsLayerTreeLayer(rl)
            parent.insertChildNode(idx, new_node)
        else:
            # No prediction node found — insert at the very top of the root
            from qgis.core import QgsLayerTreeLayer
            root.insertChildNode(0, QgsLayerTreeLayer(rl))

        self.txt_results.append(f"Errors layer added: {layer_name}")

    @staticmethod
    def _apply_symbology(layer):
        # Value  Colour                       Legend label (short form)
        # -1  → blue,  70 % opaque  (FN)
        #  0  → black, fully transparent (TN)
        #  1  → red,   70 % opaque  (FP)
        #  2  → white, 70 % opaque  (TP)
        # Alpha 178 ≈ 70 % of 255
        entries = [
            QgsPalettedRasterRenderer.Class(-1, QColor(  0,   0, 255, 178), "FN"),
            QgsPalettedRasterRenderer.Class( 0, QColor(  0,   0,   0,   0), "TN"),
            QgsPalettedRasterRenderer.Class( 1, QColor(255,   0,   0, 178), "FP"),
            QgsPalettedRasterRenderer.Class( 2, QColor(255, 255, 255, 178), "TP"),
        ]
        renderer = QgsPalettedRasterRenderer(layer.dataProvider(), 1, entries)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

    # ------------------------------------------------------------------ close

    def closeEvent(self, event):
        # Disconnect project signals to avoid dangling references
        try:
            QgsProject.instance().layersAdded.disconnect(self._refresh_layer_lists)
            QgsProject.instance().layersRemoved.disconnect(self._refresh_layer_lists)
        except Exception:
            pass
        super().closeEvent(event)
