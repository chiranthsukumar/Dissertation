from __future__ import annotations

import os
import re
import sys
import time
import threading
import uuid
from collections import deque
from pathlib import Path


def _under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__" and not _under_streamlit():
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", os.path.abspath(__file__), *sys.argv[1:]]
    sys.exit(stcli.main())

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("YOLO_VERBOSE", "False")

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import torch
from PIL import Image
from insightface.app import FaceAnalysis
from streamlit_autorefresh import st_autorefresh
from streamlit.components.v1 import html as components_html
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
from transformers import ViTForImageClassification, ViTImageProcessor
from ultralytics import YOLO

HERE = Path(__file__).parent
KNOWN_DIR = HERE / "known_faces"
KNOWN_DIR.mkdir(exist_ok=True)
CACHE_DIR = HERE / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.42         # cosine sim required to call it a known match
SESSION_DEDUP = 0.55     # cosine sim above which two detections are "same person"
DETECT_EVERY = 2         # run heavy detectors every Nth frame

# Emotion model 
EMOTION_REPO = "abhilash88/face-emotion-detection"
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NEGATIVE_EMOTIONS = {"Angry", "Fear", "Disgust"}

# YOLO 
YOLO_PATH = CACHE_DIR / "yolov8n.pt"
# COCO classes that look weapon-ish or could be used as one
WEAPON_CLASSES = {"knife", "scissors", "baseball bat", "bottle"}
PERSON_CLASS = "person"


RISK_WEIGHTS = {
    "motion":         0.10,   # fast scene motion
    "neg_emotion":    0.15,   # avg fraction of Angry/Fear/Disgust faces
    "crowd":          0.05,   # number of people detected
    "concealed_face": 0.20,   # person visible but no face detected (mask/hood)
    "weapon":         0.40,   # weapon-like object detected
    "aggressive_pose":0.20,   # arms above shoulders / fast wrist motion
    "loitering":      0.15,   # any track lingering > LOITER_SECS
    "pacing":         0.25,   # any track reversing direction repeatedly
}
# 3-tier thresholds
T_YELLOW, T_RED = 0.30, 0.60

MOTION_WINDOW = 12       # frames of optical flow to average
SCENE_HISTORY = 6        # majority-vote smoothing of the risk tier


TRACK_IOU = 0.30         # min IoU to associate a detection with an existing track
TRACK_MAX_AGE = 1.5      # seconds a track survives without being matched
LOITER_SECS = 20.0       # below this contributes 0; at LOITER_FULL contributes 1
LOITER_FULL = 60.0
PACING_WINDOW = 8.0      # seconds of history examined for direction reversals
PACING_REVERSALS = 3     # >= this many sign flips in window -> pacing flag
PACING_MIN_DX = 0.01     # ignore micro jitter per step (fraction of frame width)
PACING_MIN_SPAN = 0.10   # track must cover >=10% of frame width to count as pacing


# Cached model loaders
@st.cache_resource(show_spinner="Loading face recognition model…")
def get_model() -> FaceAnalysis:
    os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
    fa = FaceAnalysis(
        name="buffalo_s",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    fa.prepare(ctx_id=-1, det_size=(480, 480))
    return fa


@st.cache_resource(show_spinner="Loading emotion model (first run downloads ~340 MB)…")
def get_emotion_model() -> tuple[ViTImageProcessor, ViTForImageClassification]:
    processor = ViTImageProcessor.from_pretrained(EMOTION_REPO)
    model = ViTForImageClassification.from_pretrained(EMOTION_REPO)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner="Loading object detector…")
def get_yolo() -> YOLO:
    if YOLO_PATH.exists():
        return YOLO(str(YOLO_PATH))
    # Fall back to ultralytics' own cache lookup / download
    return YOLO("yolov8n.pt")


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / float(a_area + b_area - inter + 1e-6)


class SimpleTracker:
    """Tiny IoU-greedy tracker. One per FaceProcessor (state is per-session)."""

    def __init__(self, iou_thresh: float = TRACK_IOU,
                 max_age: float = TRACK_MAX_AGE) -> None:
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self._next_id = 1
        # tid -> {"bbox", "first_seen", "last_seen", "history": deque[(ts,cx,cy)]}
        self.tracks: dict[int, dict] = {}

    def update(self, detections: list[tuple], ts: float,
               frame_w: int) -> list[tuple[int, tuple]]:
        # Evict stale tracks BEFORE matching so a long-dead track can never
        # be revived by a coincidentally overlapping detection.
        stale = [tid for tid, tr in self.tracks.items()
                 if ts - tr["last_seen"] > self.max_age]
        for tid in stale:
            del self.tracks[tid]

        # Greedy IoU matching: best IoU first
        unmatched_dets = list(range(len(detections)))
        unmatched_tids = list(self.tracks.keys())
        pairs = []
        for di in unmatched_dets:
            for tid in unmatched_tids:
                pairs.append((_iou(detections[di], self.tracks[tid]["bbox"]),
                              di, tid))
        pairs.sort(key=lambda p: -p[0])
        matched_d, matched_t = set(), set()
        assignments: list[tuple[int, int]] = []
        for iou, di, tid in pairs:
            if iou < self.iou_thresh:
                break
            if di in matched_d or tid in matched_t:
                continue
            matched_d.add(di); matched_t.add(tid)
            assignments.append((di, tid))

        results: list[tuple[int, tuple]] = []
        # Update matched
        for di, tid in assignments:
            bbox = detections[di]
            cx = (bbox[0] + bbox[2]) / 2 / max(frame_w, 1)
            cy = (bbox[1] + bbox[3]) / 2
            tr = self.tracks[tid]
            tr["bbox"] = bbox
            tr["last_seen"] = ts
            tr["history"].append((ts, cx, cy))
            # Trim history to PACING_WINDOW
            while tr["history"] and ts - tr["history"][0][0] > PACING_WINDOW:
                tr["history"].popleft()
            results.append((tid, bbox))

        # Spawn new tracks for unmatched detections
        for di in range(len(detections)):
            if di in matched_d:
                continue
            tid = self._next_id; self._next_id += 1
            bbox = detections[di]
            cx = (bbox[0] + bbox[2]) / 2 / max(frame_w, 1)
            cy = (bbox[1] + bbox[3]) / 2
            self.tracks[tid] = {
                "bbox": bbox, "first_seen": ts, "last_seen": ts,
                "history": deque([(ts, cx, cy)]),
            }
            results.append((tid, bbox))

        return results

    def loitering_score(self, ts: float) -> float:
        """Max loitering across active tracks, normalized to [0,1]."""
        best = 0.0
        for tr in self.tracks.values():
            age = ts - tr["first_seen"]
            if age <= LOITER_SECS:
                continue
            score = (age - LOITER_SECS) / max(LOITER_FULL - LOITER_SECS, 1e-6)
            best = max(best, min(score, 1.0))
        return best

    def pacing_score(self) -> float:
        """1.0 if any track has >= PACING_REVERSALS horizontal direction flips
        AND the track moved across at least PACING_MIN_SPAN of the frame.
        The span check kills false positives from a stationary person whose
        bbox jitters back and forth around a fixed point."""
        for tr in self.tracks.values():
            hist = list(tr["history"])
            if len(hist) < 4:
                continue
            xs = [p[1] for p in hist]
            if max(xs) - min(xs) < PACING_MIN_SPAN:
                continue
            reversals = 0
            prev_sign = 0
            for i in range(1, len(hist)):
                dx = hist[i][1] - hist[i - 1][1]
                if abs(dx) < PACING_MIN_DX:
                    continue
                sign = 1 if dx > 0 else -1
                if prev_sign != 0 and sign != prev_sign:
                    reversals += 1
                prev_sign = sign
            if reversals >= PACING_REVERSALS:
                return 1.0
        return 0.0


def make_pose():
    """MediaPipe Pose objects are NOT thread-safe and must not be shared
    across the WebRTC worker threads. Each FaceProcessor owns its own."""
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,            # smallest pose model
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


# Inference helpers
def predict_emotion(face_bgr: np.ndarray) -> tuple[str, float]:
    if face_bgr is None or face_bgr.size == 0:
        return "Neutral", 0.0
    processor, model = get_emotion_model()
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = processor(pil, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    idx = int(torch.argmax(probs))
    return EMOTION_LABELS[idx], float(probs[idx])


def analyze_objects(img_bgr: np.ndarray) -> dict:
    """Run YOLOv8n on the frame. Returns persons + weapon detections."""
    yolo = get_yolo()
    res = yolo.predict(img_bgr, imgsz=320, conf=0.35, verbose=False)[0]
    names = res.names
    persons: list[tuple[int, int, int, int]] = []
    weapons: list[tuple[str, tuple[int, int, int, int]]] = []
    if res.boxes is None:
        return {"persons": persons, "weapons": weapons}
    for b in res.boxes:
        cls = names[int(b.cls.item())]
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        if cls == PERSON_CLASS:
            persons.append((x1, y1, x2, y2))
        elif cls in WEAPON_CLASSES:
            weapons.append((cls, (x1, y1, x2, y2)))
    return {"persons": persons, "weapons": weapons}


def analyze_pose(img_bgr: np.ndarray, pose) -> dict:
    """MediaPipe Pose. Returns flags for aggressive-posture heuristics.

    `pose` must be the per-FaceProcessor instance — never share across threads.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = pose.process(rgb)
    rgb.flags.writeable = True
    if not res.pose_landmarks:
        return {"arms_up": False, "arms_wide": False, "aggressive": False}
    lm = res.pose_landmarks.landmark
    L = mp.solutions.pose.PoseLandmark
    # y is normalized 0..1 with 0 at the top of the image, so "above" means smaller y
    l_wrist, r_wrist = lm[L.LEFT_WRIST], lm[L.RIGHT_WRIST]
    l_shldr, r_shldr = lm[L.LEFT_SHOULDER], lm[L.RIGHT_SHOULDER]
    arms_up = (l_wrist.y < l_shldr.y - 0.05) or (r_wrist.y < r_shldr.y - 0.05)
    shoulder_w = abs(l_shldr.x - r_shldr.x) + 1e-6
    arms_wide = abs(l_wrist.x - r_wrist.x) > 2.0 * shoulder_w
    return {
        "arms_up": arms_up,
        "arms_wide": arms_wide,
        "aggressive": arms_up and arms_wide,
    }


def concealed_face_count(persons: list, faces: list) -> int:
    """Persons (YOLO) whose box contains no detected face (insightface)."""
    if not persons:
        return 0
    count = 0
    face_centers = [
        ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in (f["bbox"] for f in faces)
    ]
    for px1, py1, px2, py2 in persons:
        has_face = any(
            px1 <= cx <= px2 and py1 <= cy <= py2 for cx, cy in face_centers
        )
        if not has_face:
            count += 1
    return count


def risk_score(signals: dict) -> dict:
    """Combine 0..1 signals using RISK_WEIGHTS into a single 0..1 score + tier."""
    score = 0.0
    contributions = {}
    for key, weight in RISK_WEIGHTS.items():
        s = float(np.clip(signals.get(key, 0.0), 0.0, 1.0))
        contributions[key] = s * weight
        score += contributions[key]
    score = float(np.clip(score, 0.0, 1.0))
    if score >= T_RED:
        tier, color = "RED", (0, 0, 220)
    elif score >= T_YELLOW:
        tier, color = "YELLOW", (0, 200, 220)
    else:
        tier, color = "GREEN", (40, 160, 40)
    return {"score": score, "tier": tier, "color": color,
            "signals": signals, "contrib": contributions}


# Face DB helpers 
def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\- ]", "", name).strip().replace(" ", "_")


def list_people() -> list[str]:
    return sorted(p.name for p in KNOWN_DIR.iterdir() if p.is_dir())


def load_db() -> dict[str, np.ndarray]:
    db: dict[str, np.ndarray] = {}
    for person in KNOWN_DIR.iterdir():
        if not person.is_dir():
            continue
        embs = [np.load(f) for f in person.glob("*.npy")]
        if not embs:
            continue
        avg = np.mean(np.stack(embs), axis=0)
        avg /= (np.linalg.norm(avg) + 1e-8)
        db[person.name] = avg.astype(np.float32)
    return db


def save_face(name: str, crop_bgr: np.ndarray, emb: np.ndarray) -> None:
    folder = KNOWN_DIR / safe_name(name)
    folder.mkdir(exist_ok=True)
    fid = uuid.uuid4().hex[:10]
    cv2.imwrite(str(folder / f"{fid}.jpg"), crop_bgr)
    np.save(str(folder / f"{fid}.npy"), emb.astype(np.float32))


def identify_one(emb: np.ndarray, db: dict[str, np.ndarray]) -> tuple[str | None, float]:
    if not db:
        return None, 0.0
    names = list(db.keys())
    matrix = np.stack([db[n] for n in names])
    sims = matrix @ emb
    idx = int(np.argmax(sims))
    sim = float(sims[idx])
    return (names[idx] if sim >= THRESHOLD else None), sim


# Live video processor 
class FaceProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model = get_model()
        get_emotion_model()
        get_yolo()
        self.pose = make_pose()
        self.tracker = SimpleTracker()
        self.db = load_db()
        self.last_results: list[dict] = []
        self.last_objects: dict = {"persons": [], "weapons": [], "tracks": []}
        self.last_pose: dict = {"aggressive": False, "arms_up": False}
        self._frame = 0
        self._prev_gray: np.ndarray | None = None
        self._motion_hist: deque[float] = deque(maxlen=MOTION_WINDOW)
        self._tier_hist: deque[str] = deque(maxlen=SCENE_HISTORY)
        self.scene: dict | None = None

    def reload_db(self) -> None:
        self.db = load_db()

    def _update_motion(self, img: np.ndarray) -> float:
        small = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=2,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        self._prev_gray = gray
        mag = float(np.mean(np.linalg.norm(flow, axis=2)))
        self._motion_hist.append(min(mag / 4.0, 1.0))
        return float(np.mean(self._motion_hist))

    def _draw_banner(self, img: np.ndarray, scene: dict) -> None:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 44), scene["color"], -1)
        s = scene["signals"]
        text = (
            f"{scene['tier']}  score={scene['score']:.2f}  "
            f"motion={s.get('motion', 0):.2f}  neg={s.get('neg_emotion', 0):.2f}  "
            f"weapon={s.get('weapon', 0):.0f}  hidden={s.get('concealed_face', 0):.0f}  "
            f"pose={s.get('aggressive_pose', 0):.0f}  "
            f"loiter={s.get('loitering', 0):.2f}  pace={s.get('pacing', 0):.0f}"
        )
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        self._frame += 1

        motion = self._update_motion(img)

        if self._frame % DETECT_EVERY == 0:
            # Faces + emotion
            faces = self.model.get(img)
            results = []
            for f in faces:
                x1, y1, x2, y2 = (max(0, int(v)) for v in f.bbox)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                emb = f.normed_embedding.astype(np.float32)
                name, sim = identify_one(emb, self.db)
                crop = img[y1:y2, x1:x2].copy()
                emotion, emo_conf = predict_emotion(crop)
                results.append({
                    "bbox": (x1, y1, x2, y2),
                    "name": name, "sim": sim,
                    "embedding": emb, "crop": crop,
                    "emotion": emotion, "emo_conf": emo_conf,
                })

            # YOLO objects
            try:
                objects = analyze_objects(img)
            except Exception:
                objects = {"persons": [], "weapons": []}

            # Persistent track IDs + behavioural signals
            now_ts = time.time()
            tracks = self.tracker.update(objects["persons"], now_ts, w)
            objects["tracks"] = tracks  # list of (tid, bbox)
            loiter = self.tracker.loitering_score(now_ts)
            pacing = self.tracker.pacing_score()

            try:
                pose_info = analyze_pose(img, self.pose)
            except Exception:
                pose_info = {"aggressive": False, "arms_up": False, "arms_wide": False}

            # Build combined signal vector
            n_faces = len(results)
            n_persons = max(n_faces, len(objects["persons"]))
            n_weapons = len(objects["weapons"])
            concealed = concealed_face_count(objects["persons"], results)
            neg_ratio = (
                sum(1 for r in results if r["emotion"] in NEGATIVE_EMOTIONS) / n_faces
                if n_faces else 0.0
            )

            signals = {
                "motion": motion,
                "neg_emotion": neg_ratio,
                "crowd": min(n_persons / 4.0, 1.0),
                "concealed_face": min(concealed, 2) / 2.0,
                "weapon": 1.0 if n_weapons > 0 else 0.0,
                "aggressive_pose": 1.0 if pose_info["aggressive"] else (
                    0.5 if pose_info["arms_up"] else 0.0
                ),
                "loitering": loiter,
                "pacing": pacing,
            }
            scene = risk_score(signals)

            # Smooth the tier so brief spikes don't toggle the banner.
            self._tier_hist.append(scene["tier"])
            tally = {t: list(self._tier_hist).count(t) for t in ("RED", "YELLOW", "GREEN")}
            smoothed = max(tally, key=tally.get)
            scene["tier"] = smoothed
            scene["color"] = {"RED": (0, 0, 220), "YELLOW": (0, 200, 220),
                              "GREEN": (40, 160, 40)}[smoothed]

            with self.lock:
                self.last_results = results
                self.last_objects = objects
                self.last_pose = pose_info
                self.scene = scene

        # Draw every frame so overlays track smoothly between detections.
        # Snapshot all shared state under the lock to avoid torn reads.
        with self.lock:
            results = self.last_results
            scene = self.scene
            objects_snapshot = {
                "weapons": list(self.last_objects.get("weapons", [])),
                "tracks": list(self.last_objects.get("tracks", [])),
            }

        # YOLO weapons in orange
        for cls, (x1, y1, x2, y2) in objects_snapshot["weapons"]:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(img, cls, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
        # Tracked persons in blue with stable track ID
        for tid, (x1, y1, x2, y2) in objects_snapshot["tracks"]:
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 120, 0), 1)
            cv2.putText(img, f"#{tid}", (x1, min(img.shape[0] - 5, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 120, 0), 2)

        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            color = (0, 200, 0) if r["name"] else (0, 0, 220)
            who = r["name"] if r["name"] else "Unknown"
            label = f"{who} {r['sim']:.2f} {r['emotion']}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if scene is not None:
            self._draw_banner(img, scene)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# UI FOR APP
st.set_page_config(page_title="Live Threat Detector", layout="wide")
st.title("Live Threat Detector")
st.caption(
    "Click START. Faces, emotions, weapons, and posture are analyzed in real "
    "time.."
)

get_model()
get_emotion_model()
get_yolo()

if "session_faces" not in st.session_state:
    st.session_state.session_faces = {}
if "last_alert_tier" not in st.session_state:
    st.session_state.last_alert_tier = "GREEN"
if "alerts_enabled" not in st.session_state:
    st.session_state.alerts_enabled = True
if "pending_alert" not in st.session_state:
    st.session_state.pending_alert = None


@st.dialog(" Risk Alert")
def show_alert_dialog() -> None:
    """Modal popup the user must click to dismiss. Re-renders on every
    rerun while `pending_alert` is set, so the latest info is shown."""
    a = st.session_state.get("pending_alert")
    if not a:
        return
    if a["tier"] == "RED":
        st.error(
            f"### RED — ALERT\n\n"
            f"**Risk score: {a['score']:.2f}**  ·  immediate attention"
        )
    else:
        st.warning(
            f"### YELLOW — WATCH\n\n"
            f"**Risk score: {a['score']:.2f}**  ·  elevated risk"
        )

    sig_lines = []
    pretty = {
        "weapon": "Weapon detected",
        "concealed_face": "Concealed face / mask",
        "aggressive_pose": "Aggressive posture",
        "pacing": "Pacing back and forth",
        "loitering": "Loitering",
        "neg_emotion": "Negative emotion",
        "motion": "Fast motion",
        "crowd": "Crowd",
    }
    for k, label in pretty.items():
        v = a["signals"].get(k, 0.0)
        if v > 0.05:
            sig_lines.append(f"- **{label}**: {v:.2f}")
    if sig_lines:
        st.markdown("**Active signals**")
        st.markdown("\n".join(sig_lines))

    seen_ago = max(0, int(time.time() - a["ts"]))
    st.caption(f"Triggered {seen_ago}s ago")

    if st.button("Dismiss", type="primary", width="stretch"):
        st.session_state.pending_alert = None
        st.rerun()


def fire_alert_popup(tier: str, score: float) -> None:
    """Browser desktop notification + audio beep. Different tone per tier.

    A unique nonce in the script body forces Streamlit to re-mount the
    component iframe on every alert so the JS actually runs each time.
    """
    nonce = f"{tier}-{time.time()}"
    if tier == "RED":
        title = "RED ALERT"
        body = f"Risk score {score:.2f} - immediate attention"
        freq, dur, repeats = 1320, 0.18, 3
    else:  # YELLOW
        title = "YELLOW WATCH"
        body = f"Risk score {score:.2f} - elevated risk"
        freq, dur, repeats = 880, 0.15, 1
    payload = f"""
    <script>
    // {nonce}
    (function() {{
        try {{
            if ("Notification" in window) {{
                if (Notification.permission === "granted") {{
                    new Notification({title!r}, {{ body: {body!r} }});
                }} else if (Notification.permission !== "denied") {{
                    Notification.requestPermission().then(function(p) {{
                        if (p === "granted") new Notification({title!r}, {{ body: {body!r} }});
                    }});
                }}
            }}
        }} catch (e) {{}}
        try {{
            const Ctx = window.AudioContext || window.webkitAudioContext;
            if (!Ctx) return;
            const ctx = new Ctx();
            for (let i = 0; i < {repeats}; i++) {{
                const t = ctx.currentTime + i * ({dur} + 0.08);
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.frequency.value = {freq};
                osc.type = "square";
                gain.gain.setValueAtTime(0.18, t);
                gain.gain.exponentialRampToValueAtTime(0.001, t + {dur});
                osc.connect(gain).connect(ctx.destination);
                osc.start(t); osc.stop(t + {dur});
            }}
        }} catch (e) {{}}
    }})();
    </script>
    """
    components_html(payload, height=0)


def request_notification_permission() -> None:
    """One-shot prompt: ask the browser for desktop-notification permission."""
    components_html(
        """
        <script>
        if ("Notification" in window && Notification.permission === "default") {
            Notification.requestPermission();
        }
        </script>
        """,
        height=0,
    )

video_col, side_col = st.columns([3, 1], gap="large")

with video_col:
    banner = st.empty()
    detail = st.empty()

    ctx = webrtc_streamer(
        key="threat-id",
        video_processor_factory=FaceProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    scene = (ctx.video_processor.scene
             if ctx.video_processor is not None else None)
    if scene:
        s = scene["signals"]
        line = (
            f"score {scene['score']:.2f}  · "
            f"motion {s['motion']:.2f} · "
            f"neg-emotion {s['neg_emotion']:.2f} · "
            f"crowd {s['crowd']:.2f} · "
            f"weapons {int(s['weapon'])} · "
            f"hidden faces {s['concealed_face']:.1f} · "
            f"aggressive pose {s['aggressive_pose']:.1f} · "
            f"loitering {s.get('loitering', 0):.2f} · "
            f"pacing {int(s.get('pacing', 0))}"
        )
        if scene["tier"] == "RED":
            banner.error(f"🚨 **RED — ALERT** · {line}")
        elif scene["tier"] == "YELLOW":
            banner.warning(f"⚠️ **YELLOW — WATCH** · {line}")
        else:
            banner.success(f"✅ **GREEN — NORMAL** · {line}")

        # Popup on YELLOW/RED transitions
        # Fire when entering YELLOW from GREEN, or whenever entering RED.
        # Re-fire RED alerts even from YELLOW so user always knows about RED.
        cur = scene["tier"]
        prev = st.session_state.last_alert_tier
        should_alert = (
            st.session_state.alerts_enabled
            and cur != prev
            and cur in ("YELLOW", "RED")
        )
        if should_alert:
            # Queue the modal popup (stays open until user clicks Dismiss)
            st.session_state.pending_alert = {
                "tier": cur,
                "score": scene["score"],
                "signals": dict(scene["signals"]),
                "ts": time.time(),
            }
            # One-shot browser notification + audio beep on the transition
            fire_alert_popup(cur, scene["score"])
        st.session_state.last_alert_tier = cur
    else:
        banner.info("Risk status will appear here once the camera is running.")

    # Render the dismiss-able modal if there's an active alert
    if st.session_state.alerts_enabled and st.session_state.pending_alert:
        show_alert_dialog()

with side_col:
    st.subheader("Known people")
    people = list_people()
    if people:
        for p in people:
            n = len(list((KNOWN_DIR / p).glob("*.npy")))
            st.write(f"• **{p}** _(samples: {n})_")
    else:
        st.write("_(none yet — label some faces below)_")

    st.divider()
    st.subheader("Alerts")
    st.session_state.alerts_enabled = st.toggle(
        "Pop up on YELLOW / RED",
        value=st.session_state.alerts_enabled,
        help="Browser desktop notification + audio beep when the risk tier "
             "rises. Allow notifications when your browser asks.",
    )
    if st.button("Enable browser notifications", width="stretch"):
        request_notification_permission()
        st.toast("Check the browser prompt to allow notifications.")

    st.divider()
    if st.button("Reload known faces", width="stretch"):
        if ctx.video_processor:
            ctx.video_processor.reload_db()
        st.toast("Reloaded known faces")

    if st.button("Reset session", width="stretch", type="primary"):
        st.session_state.session_faces = {}
        st.toast("Session cleared")
        st.rerun()

# Auto-rerun while the camera is playing so the table stays fresh.
labeling_in_progress = any(
    k.startswith("labeling_") and v for k, v in st.session_state.items()
)
modal_open = bool(st.session_state.get("pending_alert"))
if ctx.state.playing and not labeling_in_progress and not modal_open:
    st_autorefresh(interval=1500, key="live_refresh")

# Pull the most recent detections from the worker thread
if ctx.video_processor is not None:
    with ctx.video_processor.lock:
        current = list(ctx.video_processor.last_results)

    now = time.time()
    for r in current:
        matched_fid = None
        for fid, ex in st.session_state.session_faces.items():
            if float(np.dot(r["embedding"], ex["embedding"])) > SESSION_DEDUP:
                matched_fid = fid
                break
        if matched_fid:
            ex = st.session_state.session_faces[matched_fid]
            ex["last_seen"] = now
            ex["count"] += 1
            if r["name"] and ex["name"] != r["name"]:
                ex["name"] = r["name"]
            ex["emotion"] = r.get("emotion", ex.get("emotion", ""))
            if ex["count"] % 30 == 0:
                ex["crop"] = r["crop"]
        else:
            fid = uuid.uuid4().hex[:6]
            st.session_state.session_faces[fid] = {
                "name": r["name"],
                "embedding": r["embedding"].copy(),
                "crop": r["crop"],
                "first_seen": now, "last_seen": now,
                "count": 1, "emotion": r.get("emotion", ""),
            }

# Bottom summary table 
st.divider()
total = len(st.session_state.session_faces)
known_n = sum(1 for f in st.session_state.session_faces.values() if f["name"])
unknown_n = total - known_n

m1, m2, m3 = st.columns(3)
m1.metric("Total people seen", total)
m2.metric("Known", known_n)
m3.metric("Unknown", unknown_n)

if not st.session_state.session_faces:
    st.info("Start the camera above. Faces detected during this session will appear here.")
else:
    cols_per_row = 6
    items = list(st.session_state.session_faces.items())
    for row_start in range(0, len(items), cols_per_row):
        row = items[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (fid, face) in zip(cols, row):
            with col:
                with st.container(border=True):
                    crop = face["crop"]
                    if crop is not None and crop.size:
                        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), width="stretch")
                    if face["name"]:
                        st.markdown(f"**:green[{face['name']}]**")
                    else:
                        st.markdown(f"**:red[Unknown {fid}]**")
                    emo = face.get("emotion", "")
                    if emo:
                        st.caption(f"seen {face['count']}× · {emo}")
                    else:
                        st.caption(f"seen {face['count']}×")

                    if not face["name"]:
                        labeling_key = f"labeling_{fid}"
                        conflict_key = f"conflict_{fid}"

                        if not st.session_state.get(labeling_key):
                            if st.button("Label", key=f"open_{fid}", width="stretch"):
                                st.session_state[labeling_key] = True
                                st.rerun()
                        else:
                            people_now = list_people()
                            name_in = st.text_input(
                                "Name", key=f"name_{fid}",
                                placeholder="e.g. Alex Kim",
                                label_visibility="collapsed",
                            )
                            bcols = st.columns(2)
                            with bcols[0]:
                                save_clicked = st.button(
                                    "Save", key=f"save_{fid}",
                                    type="primary", width="stretch",
                                )
                            with bcols[1]:
                                cancel_clicked = st.button(
                                    "Cancel", key=f"cancel_{fid}", width="stretch",
                                )

                            if cancel_clicked:
                                st.session_state.pop(labeling_key, None)
                                st.session_state.pop(conflict_key, None)
                                st.rerun()

                            if save_clicked:
                                clean = safe_name(name_in or "")
                                if not clean:
                                    st.warning("Please enter a name.")
                                elif clean in people_now:
                                    st.session_state[conflict_key] = clean
                                    st.rerun()
                                else:
                                    save_face(clean, face["crop"], face["embedding"])
                                    face["name"] = clean
                                    if ctx.video_processor:
                                        ctx.video_processor.reload_db()
                                    st.session_state.pop(labeling_key, None)
                                    st.session_state.pop(conflict_key, None)
                                    st.toast(f"Saved to new folder '{clean}'")
                                    st.rerun()

                            if st.session_state.get(conflict_key):
                                existing = st.session_state[conflict_key]
                                st.warning(f"**{existing}** exists. Same person?")
                                cc1, cc2 = st.columns(2)
                                with cc1:
                                    if st.button("Yes, add", key=f"same_{fid}",
                                                 width="stretch"):
                                        save_face(existing, face["crop"], face["embedding"])
                                        face["name"] = existing
                                        if ctx.video_processor:
                                            ctx.video_processor.reload_db()
                                        st.session_state.pop(labeling_key, None)
                                        st.session_state.pop(conflict_key, None)
                                        st.toast(f"Added to '{existing}'")
                                        st.rerun()
                                with cc2:
                                    if st.button("No, rename", key=f"diff_{fid}",
                                                 width="stretch"):
                                        st.session_state.pop(conflict_key, None)
                                        st.rerun()
