#!/usr/bin/env python3
"""
GoPro camera controller - PySide6 graphical interface with live preview and
a built-in photo/video viewer.

This program uses the official "open-gopro" Python package (tested against
version 0.22.0) to communicate with a GoPro camera over WiFi/BLE or USB
(Open GoPro API), and exposes the functionality through a simple PySide6
window. The right side of the window shows either:
    - a live preview stream from the camera (decoded with OpenCV), or
    - a photo / video from the camera's file list, downloading it first if
      it is not already saved locally.

Install (one time):
    pip install open-gopro PySide6 opencv-python numpy

Usage:
    python gopro_gui.py

    1. Turn on the GoPro camera.
    2. Pick a connection mode:
       - "Wireless (BLE + WiFi)" - requires a Bluetooth adapter on this
         machine.
       - "USB (Wired)" - connect the camera to this machine with a USB
         cable first, and make sure USB connection is enabled on the
         camera (Settings > Connections > USB Connection > GoPro Connect,
         depending on camera/firmware).
    3. Click "Connect".
    4. Once connected:
       - Click "Start Preview" to stream a live low-latency preview into
         the panel on the right.
       - Or refresh the file list, select a photo/video, and click
         "View / Play Selected" (or double-click the item) to download it
         (if needed) and show/play it in the same panel.

Notes on the design:
    The GoPro library is asyncio-based (async/await), while PySide6 has its
    own event loop. To bridge the two, this program runs the asyncio event
    loop on a dedicated background thread. GUI buttons trigger the
    camera-communicating async functions from there in a thread-safe way
    (via run_coroutine_threadsafe). Results are sent back to the GUI through
    Qt signals.

    The right-hand panel is a QStackedWidget with three pages: an "idle"
    label, an image label (used for both the live preview frames and for
    viewing downloaded photos), and a QVideoWidget (used for playing
    downloaded videos via QMediaPlayer). Only one is shown at a time.

Note on locale:
    The WiFi adapter used by WirelessGoPro parses the output of system
    network CLI tools and requires an English (en_US) locale to do so. On a
    system configured with a non-English locale (e.g. hu_HU), constructing
    WirelessGoPro() would otherwise raise a RuntimeError. To avoid having to
    set this manually every time, we force it here before importing
    open_gopro. This only affects this process, not your system settings.
    This does not affect the USB (WiredGoPro) mode, but it is harmless to
    set regardless.
"""

import os

os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"

import asyncio
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QListWidget,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QMessageBox,
    QStackedWidget,
)

try:
    from open_gopro import WirelessGoPro, WiredGoPro
    from open_gopro.models import proto, streaming
    from open_gopro.models.constants import Toggle
    from open_gopro.models.constants.settings import VideoResolution, FrameRate
except ImportError:
    print("Missing 'open-gopro' package. Install it with: pip install open-gopro")
    sys.exit(1)

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


DOWNLOAD_DIR = Path("gopro_downloads")
PREVIEW_PORT = 8554

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".lrv"}

CONNECTION_MODES = {
    "Wireless (BLE + WiFi)": "wireless",
    "USB (Wired)": "wired",
}

# Preset groups control which "mode" the camera is in (photo/video/timelapse).
PRESET_GROUP_PHOTO = proto.EnumPresetGroup.PRESET_GROUP_ID_PHOTO
PRESET_GROUP_VIDEO = proto.EnumPresetGroup.PRESET_GROUP_ID_VIDEO

# A handful of common resolution / frame rate options for the UI dropdowns.
RESOLUTIONS = {
    "4K": VideoResolution.NUM_4K,
    "2.7K": VideoResolution.NUM_2_7K,
    "1080p": VideoResolution.NUM_1080,
}

FRAMERATES = {
    "240 fps": FrameRate.NUM_240_0,
    "120 fps": FrameRate.NUM_120_0,
    "60 fps": FrameRate.NUM_60_0,
    "30 fps": FrameRate.NUM_30_0,
}


class PreviewWorker(QThread):
    """Reads frames from the camera's live preview UDP feed using OpenCV and
    emits each one as a QImage. Runs on its own thread since cv2.read() is a
    blocking call."""

    frame_ready = Signal(QImage)
    error = Signal(str)

    def __init__(self, url: str):
        super().__init__()
        self._url = url
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        capture = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        if not capture.isOpened():
            self.error.emit("Could not open the preview stream with OpenCV.")
            return

        while not self._stop_requested:
            ok, frame = capture.read()
            if not ok:
                continue
            # OpenCV gives frames as BGR; Qt expects RGB.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_frame.shape
            bytes_per_line = channels * width
            image = QImage(
                rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888
            ).copy()  # copy() so the buffer survives after this loop iteration
            self.frame_ready.emit(image)

        capture.release()


class GoProController(QObject):
    """Runs in the background with its own asyncio event loop, talks to the
    camera, and notifies the GUI of events through Qt signals."""

    log_message = Signal(str)
    connected_changed = Signal(bool)
    battery_updated = Signal(str)
    media_list_ready = Signal(list)
    recording_changed = Signal(bool)
    busy_changed = Signal(bool)
    preview_started = Signal(str)  # emits the stream URL
    preview_stopped = Signal()
    media_ready_to_view = Signal(str)  # emits a local file path

    def __init__(self):
        super().__init__()
        self.gopro = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        # Runs on the background thread and keeps the asyncio loop alive.
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro):
        """Schedule a coroutine on the background event loop in a
        thread-safe way."""
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    # --- Public methods callable from the GUI buttons ---------------------

    def connect(self, mode: str):
        self._submit(self._connect(mode))

    def disconnect(self):
        self._submit(self._disconnect())

    def take_photo(self):
        self._submit(self._take_photo())

    def start_recording(self):
        self._submit(self._set_shutter(True))

    def stop_recording(self):
        self._submit(self._set_shutter(False))

    def set_resolution(self, label: str):
        self._submit(self._set_resolution(RESOLUTIONS[label]))

    def set_fps(self, label: str):
        self._submit(self._set_fps(FRAMERATES[label]))

    def refresh_battery(self):
        self._submit(self._refresh_battery())

    def refresh_media_list(self):
        self._submit(self._refresh_media_list())

    def download_file(self, filename: str):
        self._submit(self._download_file(filename))

    def download_all(self, filenames: list):
        self._submit(self._download_all(filenames))

    def start_preview(self):
        self._submit(self._start_preview())

    def stop_preview(self):
        self._submit(self._stop_preview())

    def view_file(self, filename: str):
        self._submit(self._view_file(filename))

    # --- Internal async implementations ------------------------------------

    async def _connect(self, mode: str):
        self.busy_changed.emit(True)
        try:
            if mode == "wired":
                self.log_message.emit("Connecting to the GoPro camera over USB...")
                self.gopro = WiredGoPro()
            else:
                self.log_message.emit("Connecting to the GoPro camera (BLE + WiFi)...")
                self.gopro = WirelessGoPro()
            await self.gopro.open()
            self.log_message.emit("Connected successfully!")
            self.connected_changed.emit(True)
            await self._refresh_battery()
        except Exception as exc:
            self.log_message.emit(f"Connection error: {exc}")
            self.gopro = None
            self.connected_changed.emit(False)
        finally:
            self.busy_changed.emit(False)

    async def _disconnect(self):
        if self.gopro:
            try:
                await self._stop_preview()
                await self.gopro.close()
            except Exception as exc:
                self.log_message.emit(f"Error while disconnecting: {exc}")
        self.gopro = None
        self.connected_changed.emit(False)
        self.log_message.emit("Disconnected.")

    async def _take_photo(self):
        if not self.gopro:
            return
        self.busy_changed.emit(True)
        try:
            self.log_message.emit("Switching to photo mode...")
            await self.gopro.http_command.load_preset_group(group=PRESET_GROUP_PHOTO)
            await self.gopro.http_command.set_shutter(shutter=Toggle.ENABLE)
            self.log_message.emit("Photo captured.")
        except Exception as exc:
            self.log_message.emit(f"Error while taking photo: {exc}")
        finally:
            self.busy_changed.emit(False)

    async def _set_shutter(self, start: bool):
        if not self.gopro:
            return
        try:
            if start:
                self.log_message.emit("Switching to video mode and starting recording...")
                await self.gopro.http_command.load_preset_group(group=PRESET_GROUP_VIDEO)
                await self.gopro.http_command.set_shutter(shutter=Toggle.ENABLE)
                self.log_message.emit("Recording in progress...")
            else:
                await self.gopro.http_command.set_shutter(shutter=Toggle.DISABLE)
                self.log_message.emit("Recording stopped.")
            self.recording_changed.emit(start)
        except Exception as exc:
            self.log_message.emit(f"Error while controlling recording: {exc}")

    async def _set_resolution(self, value):
        if not self.gopro:
            return
        try:
            await self.gopro.http_setting.video_resolution.set(value)
            self.log_message.emit("Resolution updated.")
        except Exception as exc:
            self.log_message.emit(f"Error while setting resolution: {exc}")

    async def _set_fps(self, value):
        if not self.gopro:
            return
        try:
            await self.gopro.http_setting.frame_rate.set(value)
            self.log_message.emit("FPS updated.")
        except Exception as exc:
            self.log_message.emit(f"Error while setting FPS: {exc}")

    async def _refresh_battery(self):
        if not self.gopro:
            return
        try:
            state = await self.gopro.http_command.get_camera_state()
            battery_pct = None
            status = getattr(state.data, "status", None)
            if status:
                # The status object stores state as key-value pairs; find
                # the battery-related entry by matching its enum name.
                for key, value in status.items():
                    key_name = getattr(key, "name", str(key))
                    if "BATTERY" in key_name.upper():
                        battery_pct = value
                        break
            if battery_pct is not None:
                self.battery_updated.emit(f"{battery_pct}%")
            else:
                self.battery_updated.emit("unknown")
        except Exception as exc:
            self.log_message.emit(f"Error while fetching status: {exc}")

    async def _refresh_media_list(self):
        if not self.gopro:
            return
        try:
            media = (await self.gopro.http_command.get_media_list()).data.files
            filenames = [item.filename for item in media]
            self.media_list_ready.emit(filenames)
            self.log_message.emit(f"Found {len(filenames)} file(s) on the camera.")
        except Exception as exc:
            self.log_message.emit(f"Error while fetching the media list: {exc}")

    async def _ensure_downloaded(self, filename: str) -> Path:
        """Downloads the given camera file to DOWNLOAD_DIR if it is not
        already there, and returns its local path."""
        local_path = DOWNLOAD_DIR / filename
        if local_path.exists():
            return local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_message.emit(f"Downloading: {filename}...")
        await self.gopro.http_command.download_file(
            camera_file=filename, local_file=local_path
        )
        self.log_message.emit(f"Downloaded to: {local_path}")
        return local_path

    async def _download_file(self, filename: str):
        if not self.gopro:
            return
        try:
            await self._ensure_downloaded(filename)
        except Exception as exc:
            self.log_message.emit(f"Error while downloading {filename}: {exc}")

    async def _download_all(self, filenames: list):
        for filename in filenames:
            await self._download_file(filename)
        self.log_message.emit("Finished downloading all files.")

    async def _view_file(self, filename: str):
        if not self.gopro:
            return
        try:
            local_path = await self._ensure_downloaded(filename)
            self.media_ready_to_view.emit(str(local_path))
        except Exception as exc:
            self.log_message.emit(f"Error while preparing {filename} for viewing: {exc}")

    async def _start_preview(self):
        if not self.gopro:
            return
        try:
            self.log_message.emit("Starting preview stream...")
            result = await self.gopro.streaming.start_stream(
                stream_type=streaming.StreamType.PREVIEW,
                options=streaming.PreviewStreamOptions(port=PREVIEW_PORT),
            )
            if not result.ok:
                self.log_message.emit(f"Failed to start preview stream: {result}")
                return
            url = self.gopro.streaming.url
            self.log_message.emit(f"Preview stream started at {url}")
            self.preview_started.emit(url)
        except Exception as exc:
            self.log_message.emit(f"Error while starting preview stream: {exc}")

    async def _stop_preview(self):
        if not self.gopro:
            return
        try:
            await self.gopro.streaming.stop_active_stream()
            self.log_message.emit("Preview stream stopped.")
        except Exception as exc:
            self.log_message.emit(f"Error while stopping preview stream: {exc}")
        finally:
            self.preview_stopped.emit()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GoPro Controller")
        self.resize(980, 640)

        self.controller = GoProController()
        self.is_recording = False
        self.preview_worker = None

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        root_layout = QHBoxLayout(self)

        # --- Left side: all of the existing controls ---
        left_panel = QWidget()
        layout = QVBoxLayout(left_panel)

        # --- Connection ---
        conn_box = QGroupBox("Connection")
        conn_layout = QVBoxLayout(conn_box)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Connection mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(CONNECTION_MODES.keys())
        mode_row.addWidget(self.mode_combo)
        conn_layout.addLayout(mode_row)

        buttons_row = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(False)
        self.status_label = QLabel("Status: not connected")
        self.battery_label = QLabel("Battery: -")
        buttons_row.addWidget(self.connect_btn)
        buttons_row.addWidget(self.disconnect_btn)
        buttons_row.addWidget(self.status_label)
        buttons_row.addWidget(self.battery_label)
        conn_layout.addLayout(buttons_row)
        layout.addWidget(conn_box)

        # --- Photo / video ---
        capture_box = QGroupBox("Photo / Video")
        capture_layout = QHBoxLayout(capture_box)
        self.photo_btn = QPushButton("Take Photo")
        self.record_btn = QPushButton("Start Recording")
        capture_layout.addWidget(self.photo_btn)
        capture_layout.addWidget(self.record_btn)
        layout.addWidget(capture_box)

        # --- Settings ---
        settings_box = QGroupBox("Settings")
        settings_layout = QHBoxLayout(settings_box)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(RESOLUTIONS.keys())
        self.resolution_apply_btn = QPushButton("Apply Resolution")
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(FRAMERATES.keys())
        self.fps_apply_btn = QPushButton("Apply FPS")
        settings_layout.addWidget(QLabel("Resolution:"))
        settings_layout.addWidget(self.resolution_combo)
        settings_layout.addWidget(self.resolution_apply_btn)
        settings_layout.addWidget(QLabel("FPS:"))
        settings_layout.addWidget(self.fps_combo)
        settings_layout.addWidget(self.fps_apply_btn)
        layout.addWidget(settings_box)

        # --- Files ---
        files_box = QGroupBox("Files on Camera")
        files_layout = QVBoxLayout(files_box)
        files_btn_layout = QHBoxLayout()
        self.refresh_files_btn = QPushButton("Refresh File List")
        self.view_selected_btn = QPushButton("View / Play Selected")
        self.download_selected_btn = QPushButton("Download Selected")
        self.download_all_btn = QPushButton("Download All")
        files_btn_layout.addWidget(self.refresh_files_btn)
        files_btn_layout.addWidget(self.view_selected_btn)
        files_btn_layout.addWidget(self.download_selected_btn)
        files_btn_layout.addWidget(self.download_all_btn)
        self.file_list = QListWidget()
        files_layout.addLayout(files_btn_layout)
        files_layout.addWidget(self.file_list)
        layout.addWidget(files_box)

        # --- Log ---
        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(log_box)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        log_layout.addWidget(self.log_view)
        layout.addWidget(log_box)

        root_layout.addWidget(left_panel, stretch=1)

        # --- Right side: live preview / media viewer ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        preview_box = QGroupBox("Live Preview / Media Viewer")
        preview_layout = QVBoxLayout(preview_box)

        self.media_stack = QStackedWidget()
        self.media_stack.setMinimumSize(480, 270)
        self.media_stack.setStyleSheet("background-color: black;")

        self.idle_label = QLabel("Nothing to show yet")
        self.idle_label.setAlignment(Qt.AlignCenter)
        self.idle_label.setStyleSheet("color: white;")

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: white;")

        self.video_widget = QVideoWidget()

        self.media_stack.addWidget(self.idle_label)   # index 0
        self.media_stack.addWidget(self.image_label)  # index 1
        self.media_stack.addWidget(self.video_widget)  # index 2

        preview_layout.addWidget(self.media_stack)

        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)

        preview_btn_layout = QHBoxLayout()
        self.start_preview_btn = QPushButton("Start Preview")
        self.stop_preview_btn = QPushButton("Stop Preview")
        self.stop_preview_btn.setEnabled(False)
        preview_btn_layout.addWidget(self.start_preview_btn)
        preview_btn_layout.addWidget(self.stop_preview_btn)
        preview_layout.addLayout(preview_btn_layout)

        right_layout.addWidget(preview_box)
        root_layout.addWidget(right_panel, stretch=1)

        if not CV2_AVAILABLE:
            self.start_preview_btn.setEnabled(False)
            self.start_preview_btn.setToolTip(
                "Live preview needs OpenCV. Install it with: pip install opencv-python"
            )

        self._set_controls_enabled(False)
        self.start_preview_btn.setEnabled(False)

    def _connect_signals(self):
        # Buttons -> controller
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        self.disconnect_btn.clicked.connect(self.controller.disconnect)
        self.photo_btn.clicked.connect(self.controller.take_photo)
        self.record_btn.clicked.connect(self._toggle_recording)
        self.resolution_apply_btn.clicked.connect(
            lambda: self.controller.set_resolution(self.resolution_combo.currentText())
        )
        self.fps_apply_btn.clicked.connect(
            lambda: self.controller.set_fps(self.fps_combo.currentText())
        )
        self.refresh_files_btn.clicked.connect(self.controller.refresh_media_list)
        self.view_selected_btn.clicked.connect(self._view_selected)
        self.download_selected_btn.clicked.connect(self._download_selected)
        self.download_all_btn.clicked.connect(self._download_all)
        self.file_list.itemDoubleClicked.connect(lambda _: self._view_selected())
        self.start_preview_btn.clicked.connect(self._on_start_preview_clicked)
        self.stop_preview_btn.clicked.connect(self.controller.stop_preview)

        # Controller -> GUI
        self.controller.log_message.connect(self._append_log)
        self.controller.connected_changed.connect(self._on_connected_changed)
        self.controller.battery_updated.connect(
            lambda pct: self.battery_label.setText(f"Battery: {pct}")
        )
        self.controller.media_list_ready.connect(self._on_media_list_ready)
        self.controller.recording_changed.connect(self._on_recording_changed)
        self.controller.busy_changed.connect(self._on_busy_changed)
        self.controller.preview_started.connect(self._on_preview_started)
        self.controller.preview_stopped.connect(self._on_preview_stopped)
        self.controller.media_ready_to_view.connect(self._on_media_ready_to_view)

    def _set_controls_enabled(self, enabled: bool):
        for widget in (
            self.photo_btn,
            self.record_btn,
            self.resolution_apply_btn,
            self.fps_apply_btn,
            self.refresh_files_btn,
            self.view_selected_btn,
            self.download_selected_btn,
            self.download_all_btn,
        ):
            widget.setEnabled(enabled)
        if CV2_AVAILABLE:
            self.start_preview_btn.setEnabled(enabled)

    def _on_connect_clicked(self):
        mode = CONNECTION_MODES[self.mode_combo.currentText()]
        self.controller.connect(mode)

    def _toggle_recording(self):
        if self.is_recording:
            self.controller.stop_recording()
        else:
            self.controller.start_recording()

    def _selected_filename(self):
        item = self.file_list.currentItem()
        if item is None:
            QMessageBox.information(self, "No selection", "Select a file from the list first.")
            return None
        return item.text()

    def _download_selected(self):
        filename = self._selected_filename()
        if filename:
            self.controller.download_file(filename)

    def _download_all(self):
        filenames = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not filenames:
            QMessageBox.information(self, "Empty list", "Refresh the file list first.")
            return
        self.controller.download_all(filenames)

    def _view_selected(self):
        filename = self._selected_filename()
        if not filename:
            return
        # Viewing a file and the live preview both use the same panel, so
        # stop any active preview stream first to avoid the two clashing.
        if self.stop_preview_btn.isEnabled():
            self.controller.stop_preview()
        self.media_player.stop()
        self.idle_label.setText(f"Loading {filename}...")
        self.media_stack.setCurrentWidget(self.idle_label)
        self.controller.view_file(filename)

    def _on_start_preview_clicked(self):
        self.media_player.stop()
        self.controller.start_preview()

    # --- Signal handlers (always run on the GUI thread) --------------------

    def _append_log(self, text: str):
        self.log_view.append(text)

    def _on_connected_changed(self, connected: bool):
        self.connect_btn.setEnabled(not connected)
        self.disconnect_btn.setEnabled(connected)
        self.mode_combo.setEnabled(not connected)
        self._set_controls_enabled(connected)
        self.status_label.setText("Status: connected" if connected else "Status: not connected")
        if not connected:
            self.battery_label.setText("Battery: -")

    def _on_media_list_ready(self, filenames: list):
        self.file_list.clear()
        self.file_list.addItems(filenames)

    def _on_recording_changed(self, recording: bool):
        self.is_recording = recording
        self.record_btn.setText("Stop Recording" if recording else "Start Recording")

    def _on_busy_changed(self, busy: bool):
        self.connect_btn.setEnabled(not busy and not self.disconnect_btn.isEnabled())

    def _on_preview_started(self, url: str):
        self.start_preview_btn.setEnabled(False)
        self.stop_preview_btn.setEnabled(True)
        self.idle_label.setText("Connecting to preview stream...")
        self.media_stack.setCurrentWidget(self.idle_label)
        self.preview_worker = PreviewWorker(url)
        self.preview_worker.frame_ready.connect(self._on_preview_frame)
        self.preview_worker.error.connect(self._on_preview_error)
        self.preview_worker.start()

    def _on_preview_stopped(self):
        self.start_preview_btn.setEnabled(True)
        self.stop_preview_btn.setEnabled(False)
        if self.preview_worker is not None:
            self.preview_worker.stop()
            self.preview_worker.wait(2000)
            self.preview_worker = None
        self.idle_label.setText("Nothing to show yet")
        self.media_stack.setCurrentWidget(self.idle_label)

    def _on_preview_frame(self, image: QImage):
        pixmap = QPixmap.fromImage(image).scaled(
            self.media_stack.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)
        self.media_stack.setCurrentWidget(self.image_label)

    def _on_preview_error(self, message: str):
        self._append_log(f"Preview error: {message}")
        self.idle_label.setText(f"Preview error:\n{message}")
        self.media_stack.setCurrentWidget(self.idle_label)

    def _on_media_ready_to_view(self, local_path: str):
        path = Path(local_path)
        suffix = path.suffix.lower()

        if suffix in IMAGE_EXTENSIONS:
            pixmap = QPixmap(str(path)).scaled(
                self.media_stack.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            if pixmap.isNull():
                self.idle_label.setText(f"Could not load image:\n{path.name}")
                self.media_stack.setCurrentWidget(self.idle_label)
            else:
                self.image_label.setPixmap(pixmap)
                self.media_stack.setCurrentWidget(self.image_label)
        elif suffix in VIDEO_EXTENSIONS:
            self.media_player.setSource(QUrl.fromLocalFile(str(path)))
            self.media_stack.setCurrentWidget(self.video_widget)
            self.media_player.play()
        else:
            self.idle_label.setText(f"Don't know how to display:\n{path.name}")
            self.media_stack.setCurrentWidget(self.idle_label)

    def closeEvent(self, event):
        self.media_player.stop()
        if self.preview_worker is not None:
            self.preview_worker.stop()
            self.preview_worker.wait(2000)
        if self.disconnect_btn.isEnabled():
            self.controller.disconnect()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()