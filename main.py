"""
hand_tracker_tasks_autotune.py
Auto-tuned, high-FPS hand tracking with ultra-stable fingertip bubbles (thumb & index).
+ Pinch detection with hysteresis (bubbles flash red + label when pinched)

- MediaPipe Tasks HandLandmarker (Apple Silicon–friendly)
- One Euro smoothing
- Camera auto-probe (FPS x resolution)
- Zero-latency capture thread (keeps newest frame)
- Processing downscale + auto-downshift when proc FPS lags
- Toggle skeleton drawing, choose num_hands
Press 'q' to quit.
"""

import os
import cv2
import time
import numpy as np
import threading
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

# ------------------ CONFIG ------------------
# If your GPU/Metal stack is flaky, uncomment to force CPU:
# os.environ["MEDIAPIPE_USE_GPU"] = "0"

DRAW_SKELETON = False       # Turn on for debug visuals (slower)
NUM_HANDS     = 2           # 1 for speed, 2 if you need both
TARGET_PROC_FPS = 45.0      # If sustained below this, we auto-downshift processing size

# Candidate camera settings to probe (ordered by desirability)
CAM_RES = [(1280, 720), (960, 540), (640, 480)]
CAM_FPS = [120, 60, 30]

# Processing pyramid: we feed a downscaled frame to the model for speed
PROC_SCALES = [1.0, 0.75, 0.5]  # 1.0 = full capture size, 0.5 = half in each dimension
PROC_DOWNSHIFT_GRACE_S = 2.0    # time under target before downshifting

# Default window size (bigger than capture)
WINDOW_INIT_W = 1280
WINDOW_INIT_H = 900
WINDOW_NAME = "Hand Tracking (Autotune) - Ultra Stable Bubbles"

# Pinch hysteresis (as fraction of palm width)
PINCH_ON_RATIO  = 0.28
PINCH_OFF_RATIO = 0.35
# --------------------------------------------

# Limit OpenCV threads to avoid contention (often helps)
try:
    cv2.setNumThreads(1)
except Exception:
    pass

# -------------- MediaPipe Tasks imports --------------
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# -------------- Model bootstrap --------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_DIR = Path.home() / ".mediapipe_models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

def ensure_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"[setup] Downloading model to {MODEL_PATH} ...")
        urlretrieve(MODEL_URL, MODEL_PATH)
        print("[setup] Model downloaded.")
    return str(MODEL_PATH)

# -------------- One Euro Filter --------------
class OneEuroFilter:
    def __init__(self, freq=60.0, min_cutoff=1.2, beta=0.03, dcutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.t_prev = None
        self.x_prev = None
        self.dx_prev = None

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te  = 1.0 / max(freq, 1e-6)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, t_now):
        x = x.astype(np.float32)
        if self.t_prev is None:
            self.t_prev = t_now
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            return x

        dt = max(t_now - self.t_prev, 1e-6)
        self.freq = 1.0 / dt
        self.t_prev = t_now

        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.dcutoff, self.freq)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat))
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

# -------------- Drawing helpers --------------
def draw_labelled_bubble(img, center_xy, radius, fill_bgr, text):
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    cv2.circle(img, (cx, cy), radius, fill_bgr, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (cx, cy), radius+2, (255,255,255), 2, lineType=cv2.LINE_AA)
    pos = (cx - radius - 6, cy - radius - 8)
    cv2.putText(img, text, (pos[0]+1, pos[1]+1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

def draw_min_skeleton(img, pts):
    lines = [(0,1),(1,2),(2,3),(3,4),
             (5,6),(6,7),(7,8),
             (9,10),(10,11),(11,12),
             (13,14),(14,15),(15,16),
             (17,18),(18,19),(19,20),
             (0,5),(5,9),(9,13),(13,17)]
    for a,b in lines:
        cv2.line(img, pts[a], pts[b], (180,180,180), 2, cv2.LINE_AA)

def draw_pinch_label(img, midpoint_xy, text="Pinch detected"):
    x, y = int(round(midpoint_xy[0])), int(round(midpoint_xy[1]))
    # Slight shadow for readability
    cv2.putText(img, text, (x+1, y-21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y-22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

# -------------- Capture thread --------------
class LatestFrameCapture:
    def __init__(self, src=0, backend=cv2.CAP_AVFOUNDATION):
        self.cap = cv2.VideoCapture(src, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def set(self, prop, value):
        return self.cap.set(prop, value)

    def get(self, prop):
        return self.cap.get(prop)

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            f = None if self.frame is None else self.frame.copy()
        return f

    def release(self):
        self.stopped = True
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        self.cap.release()

# -------------- Camera auto-probe --------------
def try_cam_settings(cap, w, h, fps, sample_time=0.8):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    frames = 0
    start = time.time()
    while time.time() - start < sample_time:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        frames += 1
    measured = frames / max(time.time() - start, 1e-6)
    return measured

def pick_best_camera_combo():
    temp = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not temp.isOpened():
        temp = cv2.VideoCapture(0)

    best = None
    best_score = -1.0
    for (w, h) in CAM_RES:
        for fps in CAM_FPS:
            meas = try_cam_settings(temp, w, h, fps)
            score = meas - (w*h)/ (1920*1080*2.0)  # prefer FPS, lightly penalize big res
            if score > best_score:
                best_score = score
                best = (w, h, fps, meas)
    temp.release()
    return best  # (w,h,fps,measured_fps)

# -------------- HandLandmarker setup --------------
def create_hand_landmarker():
    model_path = ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.75,
        min_hand_presence_confidence=0.55,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    return mp_vision.HandLandmarker.create_from_options(options)

def handed_label(handedness_list):
    if not handedness_list:
        return "Right"
    name = handedness_list[0].category_name
    return "Left" if name.lower().startswith("left") else "Right"

# -------------- Main --------------
def main():
    print("[autotune] Probing camera for best FPS/res…")
    w, h, req_fps, measured = pick_best_camera_combo()
    print(f"[autotune] Chosen capture: {w}x{h} @ target {req_fps} (measured ~{measured:.1f} FPS)")

    cap = LatestFrameCapture(0).start()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, req_fps)

    detector = create_hand_landmarker()

    # Make the window big by default (resizable)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_INIT_W, WINDOW_INIT_H)

    print("Hand Tracking Started! Press 'q' to quit.")
    print(f"Ultra-stable thumb & index bubbles | num_hands={NUM_HANDS} | skeleton={DRAW_SKELETON}")

    # Filters per (hand_index, "thumb"/"index")
    filters = defaultdict(lambda: OneEuroFilter(freq=60.0, min_cutoff=1.2, beta=0.03, dcutoff=1.0))
    last_seen = {}
    STALE_RESET_S = 0.4

    # Landmarks
    LID_WRIST      = 0
    LID_INDEX_MCP  = 5
    LID_PINKY_MCP  = 17
    LID_THUMB_TIP  = 4
    LID_INDEX_TIP  = 8

    # Colors
    COLORS = {
        "Left":  ((255,   0,   0), (255, 120,   0)),  # thumb, index
        "Right": ((  0,   0, 255), (  0, 120, 255)),
    }
    RED = (0, 0, 255)

    # Pinch state per hand index (hysteresis)
    pinch_on = {}   # hi -> bool

    # Processing scale selection
    proc_scale_idx = 0
    below_target_since = None

    prev_disp_time = time.time()
    prev_proc_time = time.time()
    disp_fps = 0.0
    proc_fps = 0.0
    prev_ts_ms = 0

    try:
        while True:
            frame_bgr = cap.read()
            if frame_bgr is None:
                time.sleep(0.001)
                continue

            # Display FPS
            now_t = time.time()
            disp_fps = 0.9 * disp_fps + 0.1 * (1.0 / max(now_t - prev_disp_time, 1e-6))
            prev_disp_time = now_t

            frame_bgr = cv2.flip(frame_bgr, 1)
            H, W = frame_bgr.shape[:2]

            # Downscale for processing
            scale = PROC_SCALES[proc_scale_idx]
            if scale != 1.0:
                proc_w, proc_h = int(W * scale), int(H * scale)
                proc_rgb = cv2.cvtColor(cv2.resize(frame_bgr, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            else:
                proc_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                proc_h, proc_w = H, W

            # Build MP Image with monotonic timestamp
            ts_ms = int(time.time() * 1000)
            if ts_ms <= prev_ts_ms:
                ts_ms = prev_ts_ms + 1
            prev_ts_ms = ts_ms
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=proc_rgb)

            # Inference
            result = detector.detect_for_video(mp_image, ts_ms)
            proc_fps = 0.9 * proc_fps + 0.1 * (1.0 / max(time.time() - prev_proc_time, 1e-6))
            prev_proc_time = time.time()

            # Auto-downshift if sustained below target
            if proc_fps < TARGET_PROC_FPS:
                if below_target_since is None:
                    below_target_since = time.time()
                elif (time.time() - below_target_since) > PROC_DOWNSHIFT_GRACE_S:
                    if proc_scale_idx < len(PROC_SCALES) - 1:
                        proc_scale_idx += 1
                        print(f"[autotune] Proc FPS low ({proc_fps:.1f}). Downshifting to scale={PROC_SCALES[proc_scale_idx]:.2f}")
                    below_target_since = None
            else:
                below_target_since = None

            # Draw & Pinch detection
            if result.hand_landmarks:
                # Optional skeleton
                if DRAW_SKELETON:
                    for lms in result.hand_landmarks:
                        P = [(int(l.x * W), int(l.y * H)) for l in lms]
                        draw_min_skeleton(frame_bgr, P)

                for hi, (lms, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
                    label = handed_label(handedness)

                    # Extract points (display space)
                    thumb = lms[LID_THUMB_TIP]; index = lms[LID_INDEX_TIP]
                    index_mcp = lms[LID_INDEX_MCP]; pinky_mcp = lms[LID_PINKY_MCP]

                    thumb_px = np.array([thumb.x * W, thumb.y * H], dtype=np.float32)
                    index_px = np.array([index.x * W, index.y * H], dtype=np.float32)
                    index_mcp_px = np.array([index_mcp.x * W, index_mcp.y * H], dtype=np.float32)
                    pinky_mcp_px = np.array([pinky_mcp.x * W, pinky_mcp.y * H], dtype=np.float32)

                    # Smoothing
                    t_now = time.perf_counter()
                    for name, pt in (("thumb", thumb_px), ("index", index_px)):
                        key = (hi, name)
                        last_t = last_seen.get(key)
                        if last_t is None or (t_now - last_t) > STALE_RESET_S:
                            filters[key] = OneEuroFilter(freq=60.0, min_cutoff=1.2, beta=0.03, dcutoff=1.0)
                        last_seen[key] = t_now
                    thumb_s = filters[(hi, "thumb")](thumb_px, t_now)
                    index_s = filters[(hi, "index")](index_px, t_now)

                    # --- Pinch detection (hysteresis) ---
                    palm_width = float(np.linalg.norm(index_mcp_px - pinky_mcp_px)) + 1e-6
                    dist_tips  = float(np.linalg.norm(thumb_s - index_s))

                    is_on = pinch_on.get(hi, False)
                    if not is_on and dist_tips <= PINCH_ON_RATIO * palm_width:
                        is_on = True
                    elif is_on and dist_tips >= PINCH_OFF_RATIO * palm_width:
                        is_on = False
                    pinch_on[hi] = is_on

                    # Colors
                    thumb_color, index_color = COLORS.get(label, ((80,80,255),(80,255,80)))
                    if is_on:
                        thumb_color = index_color = (0, 0, 255)  # bright red

                    # Draw bubbles
                    draw_labelled_bubble(frame_bgr, thumb_s, 20, thumb_color, f"{label} Thumb")
                    draw_labelled_bubble(frame_bgr, index_s, 20, index_color, f"{label} Index")

                    # Pinch label at midpoint
                    if is_on:
                        midpoint = (thumb_s + index_s) / 2.0
                        draw_pinch_label(frame_bgr, midpoint, "Pinch detected")

            # HUD
            cv2.putText(frame_bgr, f"Cam: {W}x{H} @ ~{disp_fps:.0f} fps", (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 240, 90), 2, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"Proc: scale {PROC_SCALES[proc_scale_idx]:.2f}  ~{proc_fps:.0f} fps", (16, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 240, 90), 2, cv2.LINE_AA)
            cv2.putText(frame_bgr, "Press 'Q' to quit", (16, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame_bgr)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                print("Exiting…")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
