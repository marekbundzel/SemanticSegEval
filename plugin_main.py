# plugin_main.py
import os
from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon


class SemanticSegEvalPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dialog = None

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
        self.action = QAction(icon, "Semantic Segmentation Evaluator", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToRasterMenu("&Semantic Seg Eval", self.action)

    def unload(self):
        self.iface.removePluginRasterMenu("&Semantic Seg Eval", self.action)
        self.iface.removeToolBarIcon(self.action)
        if self.dialog:
            self.dialog.close()

    def run(self):
        from .dialog_main import SemanticSegEvalDialog
        if self.dialog is None or not self.dialog.isVisible():
            self.dialog = SemanticSegEvalDialog(self.iface)
        self.dialog.show()
        self.dialog.raise_()
