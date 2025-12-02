'''
using a single csi camera, yolo, and a manifest set in downloads
should start scanning immediately and jumps to view order when done
'''

import sys, os, time
from pathlib import Path
from datetime import datetime

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QStackedWidget, QTextEdit, QScrollArea, QGridLayout)

# ===================== simple manifest loader =====================
def load_manifest():
    manifest_path = "/home/jetson/Documents/barcodes.txt"

    print("#loading manifest from documents folder...")

    if not os.path.exists(manifest_path):
        print("#manifest missing, making empty list")
        return []

    txt = Path(manifest_path).read_text(encoding="utf-8", errors="ignore")
    parts = [p.strip() for p in txt.split() if p.strip()]
    uniq = []
    seen = set()
    for p in parts:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq

# ===================== view order screen =====================
class ViewOrderScreen(QWidget):
    #this screen lists finished scan summary
    return_to_welcome = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        title = QLabel("View Orders")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 36))
        title.setStyleSheet("color:#0c2340;background-color:#f15a22;font-weight:bold;")
        layout.addWidget(title)

        #scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        self.container = QWidget()
        self.scroll_area.setWidget(self.container)

        self.grid_layout = QGridLayout(self.container)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.grid_layout.setHorizontalSpacing(20)
        self.grid_layout.setVerticalSpacing(5)

        #column headers
        headers = ["Trailer", "Archway", "Start", "End", "Duration", "Scanned"]
        for col, h in enumerate(headers):
            lbl = QLabel(h)
            lbl.setFont(QFont("Arial", 12, QFont.Bold))
            lbl.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(lbl, 0, col)

        self._next_row = 1

        self.status = QLabel("Press X to restart")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(self.status)

    def add_order(self, start_time, end_time, scanned_count):
        duration = end_time - start_time
        archway = "Archway 1"
        trailer = "Trailer 01"  #hardcoded placeholder for class demo

        start_str = start_time.strftime("%H:%M:%S")
        end_str = end_time.strftime("%H:%M:%S")
        duration_str = str(duration).split(".")[0]

        values = [trailer, archway, start_str, end_str, duration_str, str(scanned_count)]

        for col, val in enumerate(values):
            lbl = QLabel(val)
            lbl.setFont(QFont("Arial", 11))
            lbl.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(lbl, self._next_row, col)

        self._next_row += 1

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_X, Qt.Key_C):
            python = sys.executable
            os.execv(python, [python] + sys.argv)

# ===================== barcode reader worker =====================
class barcodeReader(QThread):
    log = pyqtSignal(str)
    decoded = pyqtSignal(str)
    finished_all = pyqtSignal()

    def __init__(self, manifest_codes):
        super().__init__()
        self.codes = [c.strip() for c in manifest_codes]
        self._found = set()
        self._stop = False

        #scan timing
        self.fps_delay = 0.2

    def stop(self):
        self._stop = True

    def _make_pipeline(self):
        #basic csi pipeline like before
        return (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=5/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
        )

    def run(self):
        #lazy import stuff
        try:
            from ultralytics import YOLO
            from PIL import Image, ImageOps
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
            from pyzbar.pyzbar import decode as zbar_decode
            import numpy as np
        except Exception as e:
            self.log.emit(f"#import fail: {e}")
            return

        Gst.init(None)
        pipeline_str = self._make_pipeline()
        pipeline = Gst.parse_launch(pipeline_str)
        appsink = pipeline.get_by_name("sink")
        pipeline.set_state(Gst.State.PLAYING)

        self.log.emit("#loading yolo model...")
        model = YOLO("my_model.pt")

        total = len(self.codes)
        self.log.emit(f"#expecting {total} codes...")

        while not self._stop:
            #grab frame
            sample = appsink.emit("pull-sample")
            if sample is None:
                time.sleep(self.fps_delay)
                continue

            buf = sample.get_buffer()
            caps = sample.get_caps()
            w = caps.get_structure(0).get_value("width")
            h = caps.get_structure(0).get_value("height")

            ok, map_info = buf.map(Gst.MapFlags.READ)
            if not ok:
                time.sleep(self.fps_delay)
                continue

            frame = None
            try:
                frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((h, w, 3))
            finally:
                buf.unmap(map_info)

            if frame is None:
                time.sleep(self.fps_delay)
                continue

            #pillow rgb
            img_rgb = Image.fromarray(frame[:, :, ::-1], mode="RGB")

            #run yolo
            res = model.predict(img_rgb, conf=0.25, iou=0.45, verbose=False)
            if not res or res[0].boxes is None:
                self.log.emit("no barcodes read")
                time.sleep(self.fps_delay)
                continue

            boxes = res[0].boxes.xyxy.cpu().numpy().astype(int)
            decoded_vals = []

            for (x1, y1, x2, y2) in boxes:
                crop = img_rgb.crop((x1, y1, x2, y2))
                gray = ImageOps.grayscale(crop)
                out = zbar_decode(gray)
                for r in out:
                    try:
                        v = r.data.decode("utf-8", errors="ignore")
                        if v not in decoded_vals:
                            decoded_vals.append(v)
                    except:
                        pass

            if decoded_vals:
                for v in decoded_vals:
                    self.decoded.emit(v)
                    if v in self.codes:
                        self._found.add(v)
                        self.log.emit(f"{v} is loaded")
                    else:
                        self.log.emit(f"{v} not part of shipment")

                if len(self._found) >= total:
                    self.log.emit("all codes scanned")
                    self.finished_all.emit()
                    break
            else:
                self.log.emit("no barcodes read")

            time.sleep(self.fps_delay)

        pipeline.set_state(Gst.State.NULL)

# ===================== scan screen =====================
class ScanScreen(QWidget):
    #this is the simple live scan log screen
    scan_complete = pyqtSignal(int, datetime, datetime)

    def __init__(self, manifest):
        super().__init__()
        self.start_time = datetime.now()
        self.manifest = list(manifest)

        lay = QVBoxLayout(self)
        t = QLabel("Scanning...")
        t.setAlignment(Qt.AlignCenter)
        t.setFont(QFont("Arial", 36))
        t.setStyleSheet("color:#0c2340;background-color:#f15a22;font-weight:bold;")
        lay.addWidget(t)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log)

        #start worker
        self.worker = barcodeReader(self.manifest)
        self.worker.log.connect(self._log)
        self.worker.decoded.connect(self._log)
        self.worker.finished_all.connect(self._done)
        self.worker.start()

    def _log(self, msg):
        self.log.append(msg)

    def _done(self):
        end = datetime.now()
        count = len(self.manifest)
        self.scan_complete.emit(count, self.start_time, end)

# ===================== main window =====================
class MainWindow(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("final project")
        self.showFullScreen()  #full screen like you wanted

        #load manifest
        manifest = load_manifest()

        #create screens
        self.scan = ScanScreen(manifest)
        self.view = ViewOrderScreen()

        self.scan.scan_complete.connect(self._on_scan_done)

        self.addWidget(self.scan)
        self.addWidget(self.view)

        self.setCurrentIndex(0)

    def _on_scan_done(self, count, start, end):
        #casual: when done, add row + switch screens
        self.view.add_order(start, end, count)
        self.setCurrentIndex(1)

# ===================== entry =====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
