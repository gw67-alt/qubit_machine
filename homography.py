import sys
import time
import cv2
import numpy as np # For np.mean
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStatusBar
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QObject
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg
from collections import deque

# --- OpenCV Configuration (from previous script) ---
MIN_MATCH_COUNT = 10
LOWE_RATIO_TEST = 0.10
KEY_TO_CYCLE_QT = Qt.Key_N
KEY_TO_QUIT_QT = Qt.Key_Q

# --- Chart Configuration ---
MAX_CHART_POINTS = 100  # Number of data points to display on the chart
MOVING_AVG_WINDOW = 100  # Window size for the moving average

# --- State Management Object (Same as before) ---
class AppState(QObject):
    state_changed = pyqtSignal(int)
    capture_reference_requested = pyqtSignal()
    reset_requested = pyqtSignal()

    STATE_WAITING_FOR_REFERENCE = 0
    STATE_TRACKING = 1

    def __init__(self):
        super().__init__()
        self._current_state = self.STATE_WAITING_FOR_REFERENCE

    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, value):
        if self._current_state != value:
            self._current_state = value
            self.state_changed.emit(value)

    def request_capture_reference(self):
        self.capture_reference_requested.emit()

    def request_reset(self):
        self.reset_requested.emit()

app_state = AppState()

# --- OpenCV Processing Thread (Same as before) ---
class OpenCVThread(QThread):
    frame_ready = pyqtSignal(QImage)
    matches_count_ready = pyqtSignal(int)
    status_message = pyqtSignal(str)

    def __init__(self, app_state_ref):
        super().__init__()
        self.running = False
        self.app_state = app_state_ref

        self.reference_frame = None
        self.reference_kp = None
        self.reference_des = None
        self.orb = None
        self.bf_matcher = None
        self._capture_next_frame_as_reference = False


        self.app_state.capture_reference_requested.connect(self.prepare_for_reference_capture)
        self.app_state.reset_requested.connect(self.reset_reference)


    def initialize_features(self):
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def prepare_for_reference_capture(self):
        self._capture_next_frame_as_reference = True
        # self.status_message.emit("Thread: Reference capture requested.") # Debug

    def reset_reference(self):
        self.reference_frame = None
        self.reference_kp = None
        self.reference_des = None
        self._capture_next_frame_as_reference = False
        self.app_state.current_state = AppState.STATE_WAITING_FOR_REFERENCE # This will trigger state_changed signal
        # self.status_message.emit("Thread: Reference reset.") # Debug

    def run(self):
        self.running = True
        self.initialize_features()
        # self._capture_next_frame_as_reference = False # Already set in __init__ and reset_reference

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_message.emit("Error: Cannot open camera.")
            self.running = False
            return

        # Initial status message based on state is handled by MainWindow via AppState.state_changed
        # if self.app_state.current_state == AppState.STATE_WAITING_FOR_REFERENCE:
        #      self.status_message.emit("Aim camera and press 'N' to capture reference.")


        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.status_message.emit("Error: Can't receive frame.")
                time.sleep(0.5)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_ready.emit(qt_image.copy())

            num_good_matches = 0

            if self._capture_next_frame_as_reference:
                # self.status_message.emit("Thread: Capturing reference frame now.") # Debug
                self.reference_frame = frame.copy()
                self.reference_kp, self.reference_des = self.orb.detectAndCompute(self.reference_frame, None)

                if self.reference_des is None or len(self.reference_kp) < MIN_MATCH_COUNT:
                    # self.status_message.emit(f"Thread: Not enough features in reference ({len(self.reference_kp) if self.reference_kp else 'None'} kp). Try again.") # Debug
                    self.reference_frame = None
                    # No state change here, let MainWindow handle it based on failed capture if needed
                    # Or, we could emit a specific "capture_failed" signal
                    self.app_state.current_state = AppState.STATE_WAITING_FOR_REFERENCE # Stay in waiting
                else:
                    # self.status_message.emit(f"Thread: Reference captured ({len(self.reference_kp)} pts). Tracking...") # Debug
                    self.app_state.current_state = AppState.STATE_TRACKING
                self._capture_next_frame_as_reference = False


            if self.app_state.current_state == AppState.STATE_TRACKING and self.reference_frame is not None:
                current_kp, current_des = self.orb.detectAndCompute(frame, None)

                if current_des is not None and len(current_des) > 0 and self.reference_des is not None:
                    all_matches = self.bf_matcher.knnMatch(self.reference_des, current_des, k=2)
                    good_matches = []
                    for m_arr in all_matches:
                        if len(m_arr) == 2:
                            m, n = m_arr
                            if m.distance < LOWE_RATIO_TEST * n.distance:
                                good_matches.append(m)
                    num_good_matches = len(good_matches)

                    if num_good_matches >= MIN_MATCH_COUNT:
                        src_pts = np.float32([self.reference_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([current_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 150.0)
                        if H is None:
                             num_good_matches = -1
            self.matches_count_ready.emit(num_good_matches)
            self.msleep(30)

        cap.release()
        self.status_message.emit("Camera released.")


    def stop(self):
        self.running = False
        self.wait()

# --- Main Application Window (Updated) ---
class MainWindow(QMainWindow):
    def __init__(self, app_state_ref):
        super().__init__()
        self.app_state = app_state_ref
        self.setWindowTitle("OpenCV Homography with Qt5 Chart (Raw & Average Matches)")
        self.setGeometry(100, 100, 1000, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        video_layout = QVBoxLayout()
        self.video_label = QLabel("Initializing Camera...")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid black;")
        video_layout.addWidget(self.video_label)
        main_layout.addLayout(video_layout, 2)

        controls_chart_layout = QVBoxLayout()
        self.match_chart_widget = pg.PlotWidget()
        self.match_chart_widget.setBackground('w')
        self.match_chart_widget.setTitle("Feature Matches", color="k", size="12pt")
        self.match_chart_widget.setLabel('left', 'Match Count', color='k')
        self.match_chart_widget.setLabel('bottom', 'Time (frames)', color='k')
        self.match_chart_widget.showGrid(x=True, y=True)
        self.match_chart_widget.addLegend() # Add legend support

        # Plot for Raw Matches
        self.raw_match_data_line = self.match_chart_widget.plot(pen=pg.mkPen('b', width=2), name="Raw Matches")
        self.raw_match_history = deque(maxlen=MAX_CHART_POINTS)

        # Plot for Average Matches
        self.avg_match_data_line = self.match_chart_widget.plot(pen=pg.mkPen('r', width=2, style=Qt.DashLine), name=f"Avg Matches (Win: {MOVING_AVG_WINDOW})")
        self.avg_match_history = deque(maxlen=MAX_CHART_POINTS)

        self.time_points = deque(maxlen=MAX_CHART_POINTS)
        self.current_time_step = 0

        controls_chart_layout.addWidget(self.match_chart_widget)
        main_layout.addLayout(controls_chart_layout, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.opencv_thread = OpenCVThread(self.app_state)
        self.opencv_thread.frame_ready.connect(self.update_video_frame)
        self.opencv_thread.matches_count_ready.connect(self.update_matches_chart)
        self.opencv_thread.status_message.connect(self.show_status_message) # Connect this
        self.app_state.state_changed.connect(self.on_state_changed_gui)

        self.opencv_thread.start()
        self.on_state_changed_gui(self.app_state.current_state) # Set initial message

    def update_video_frame(self, q_image):
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def update_matches_chart(self, raw_count):
        actual_count = raw_count if raw_count >= 0 else 0 # Use 0 for plotting if homography failed

        self.raw_match_history.append(actual_count)
        self.time_points.append(self.current_time_step)
        self.current_time_step += 1

        # Calculate moving average
        current_avg = 0
        if len(self.raw_match_history) > 0:
            # Take up to MOVING_AVG_WINDOW last elements for averaging
            avg_window_data = list(self.raw_match_history)[-MOVING_AVG_WINDOW:]
            if avg_window_data: # Ensure not empty
                current_avg = np.mean(avg_window_data)

        self.avg_match_history.append(current_avg)

        # Ensure all deques are aligned for plotting by using time_points as the master x-axis
        # Deques handle their own maxlen, so direct list conversion is fine.
        current_time_points = list(self.time_points)
        current_raw_history = list(self.raw_match_history)
        current_avg_history = list(self.avg_match_history)

        self.raw_match_data_line.setData(current_time_points, current_raw_history)
        self.avg_match_data_line.setData(current_time_points, current_avg_history)


    def show_status_message(self, message):
        self.status_bar.showMessage(message)
        # print(f"GUI Status: {message}") # Optional: for debugging GUI-side status updates

    def on_state_changed_gui(self, state):
        if state == AppState.STATE_WAITING_FOR_REFERENCE:
            self.show_status_message("STATE: Waiting for Reference. Aim and press 'N'.")
        elif state == AppState.STATE_TRACKING:
            self.show_status_message("STATE: Tracking. Press 'N' to reset reference.")

    def keyPressEvent(self, event):
        key = event.key()
        if key == KEY_TO_QUIT_QT:
            self.close()
        elif key == KEY_TO_CYCLE_QT:
            if self.app_state.current_state == AppState.STATE_WAITING_FOR_REFERENCE:
                self.show_status_message("GUI: Requesting reference capture...")
                self.app_state.request_capture_reference()
            elif self.app_state.current_state == AppState.STATE_TRACKING:
                self.show_status_message("GUI: Requesting reset...")
                self.app_state.request_reset()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.show_status_message("Closing application...")
        self.opencv_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # pg.setConfigOptions(antialias=True) # Uncomment for smoother lines
    main_window = MainWindow(app_state)
    main_window.show()
    sys.exit(app.exec_())
