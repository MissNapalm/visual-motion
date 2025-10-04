
"""
hand_tracker_tasks.py
Ultra-stable fingertip bubbles (thumb & index) for both hands using MediaPipe **Tasks** HandLandmarker.
- One Euro filter per fingertip (sticky when still, responsive when moving)
- Auto-downloads the .task model if missing
- Apple Silicon–friendly (no old calculator graph crashes)

Run:
  python hand_tracker_tasks.py
Press 'q' to quit.
"""

import os
import time
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

# ---- Optional: force CPU if your Metal/GL stack is quirky ----
# os.environ["MEDIAPIPE_USE_GPU"] = "0"

# ---- MediaPipe Tasks imports ----
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ========== Model bootstrap ==========
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_DIR = Path.home() / ".mediapipe_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

def ensure_model():
    if not MODEL_PATH.exists():
        print(f"[setup] Downloading model to {MODEL_PATH} ...")
        urlretrieve(MODEL_URL, MODEL_PATH)
        print("[setup] Model downloaded.")
    return str(MODEL_PATH)

# ========== One Euro Filter ==========
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

# ========== Drawing helpers ==========
def draw_labelled_bubble(img, center_xy, radius, fill_bgr, text):
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    cv2.circle(img, (cx, cy), radius, fill_bgr, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (cx, cy), radius+2, (255,255,255), 2, lineType=cv2.LINE_AA)
    pos = (cx - radius - 6, cy - radius - 8)
    cv2.putText(img, text, (pos[0]+1, pos[1]+1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

# ========== Setup HandLandmarker (Tasks API) ==========
def create_hand_landmarker():
    model_path = ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.75,
        min_hand_presence_confidence=0.55,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    return mp_vision.HandLandmarker.create_from_options(options)

# ========== Main ==========
def main():
    detector = create_hand_landmarker()

    # macOS best backend; harmless elsewhere. Try AVFOUNDATION then default.
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Hand Tracking Started! Press 'q' to quit.")
    print("Ultra-stable thumb & index bubbles per hand (Tasks API).")

    # Filters per (hand_index, "thumb"/"index")
    filters = defaultdict(lambda: OneEuroFilter(freq=60.0, min_cutoff=1.2, beta=0.03, dcutoff=1.0))
    last_seen = {}
    STALE_RESET_S = 0.4

    # landmark indices
    LID_THUMB_TIP = 4
    LID_INDEX_TIP = 8

    # Colors per hand label
    COLORS = {
        "Left":  ((255,   0,   0), (255, 120,   0)),  # thumb, index
        "Right": ((  0,   0, 255), (  0, 120, 255)),
    }

    def handed_label(handedness_list):
        # Each is a list of Category; pick the top one.
        if not handedness_list:
            return "Right"
        name = handedness_list[0].category_name
        return "Left" if name.lower().startswith("left") else "Right"

    prev_ts_ms = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Failed to capture frame")
                break

            frame_bgr = cv2.flip(frame_bgr, 1)
            h, w = frame_bgr.shape[:2]

            # Build MediaPipe Image with monotonic timestamp
            ts_ms = int(time.time() * 1000)
            if ts_ms <= prev_ts_ms:
                ts_ms = prev_ts_ms + 1
            prev_ts_ms = ts_ms

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            result = detector.detect_for_video(mp_image, ts_ms)

            # Draw minimal skeleton (for context)
            if result.hand_landmarks:
                for lms in result.hand_landmarks:
                    P = [(int(l.x * w), int(l.y * h)) for l in lms]
                    def seg(a, b):
                        cv2.line(frame_bgr, P[a], P[b], (180, 180, 180), 2, cv2.LINE_AA)
                    # minimal hand mesh
                    for a,b in [(0,1),(1,2),(2,3),(3,4),
                                (5,6),(6,7),(7,8),
                                (9,10),(10,11),(11,12),
                                (13,14),(14,15),(15,16),
                                (17,18),(18,19),(19,20),
                                (0,5),(5,9),(9,13),(13,17)]:
                        seg(a,b)

                # Bubbles with filtering
                for hi, (lms, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
                    label = handed_label(handedness)

                    thumb = lms[LID_THUMB_TIP]
                    index = lms[LID_INDEX_TIP]

                    thumb_px = np.array([thumb.x * w, thumb.y * h], dtype=np.float32)
                    index_px = np.array([index.x * w, index.y * h], dtype=np.float32)

                    t_now = time.perf_counter()

                    # Reset filters if stale (prevents teleport on occlusion)
                    for name, pt in (("thumb", thumb_px), ("index", index_px)):
                        key = (hi, name)
                        last_t = last_seen.get(key)
                        if last_t is None or (t_now - last_t) > STALE_RESET_S:
                            filters[key] = OneEuroFilter(freq=60.0, min_cutoff=1.2, beta=0.03, dcutoff=1.0)
                        last_seen[key] = t_now

                    thumb_s = filters[(hi, "thumb")](thumb_px, t_now)
                    index_s = filters[(hi, "index")](index_px, t_now)

                    thumb_color, index_color = COLORS.get(label, ((80,80,255),(80,255,80)))
                    draw_labelled_bubble(frame_bgr, thumb_s, 20, thumb_color, f"{label} Thumb")
                    draw_labelled_bubble(frame_bgr, index_s, 20, index_color, f"{label} Index")

            cv2.putText(frame_bgr, "Press 'Q' to quit", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 240, 90), 2, cv2.LINE_AA)
            cv2.imshow("Hand Tracking (Tasks API) - Ultra Stable Bubbles", frame_bgr)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                print("Exiting…")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # detector doesn't need explicit close

if __name__ == "__main__":
    main()

