from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap, QImage


class Display(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(True)

    def on_image_received(self, qimage):
        pixmap = QPixmap.fromImage(qimage)
        self.setPixmap(pixmap)
