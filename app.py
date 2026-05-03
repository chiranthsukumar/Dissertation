"""Live face identifier — webcam stream with automatic detection.

Uses InsightFace's pretrained `buffalo_s` pack (SCRFD detector + ArcFace MBF
recognition). Each known person is a folder under `known_faces/<name>/` with
`.jpg` crops and `.npy` embeddings.

Also overlays per-face **emotion** (HSEmotion AffectNet ONNX) and renders a
scene-level **NORMAL / SUSPICIOUS** banner driven by motion intensity,
negative-emotion ratio across visible faces, and people count.

Run it either way:
  python app.py            (auto-launches Streamlit and opens your browser)
  streamlit run app.py
"""
from __future__ import annotations

import collections
import os
import re
import sys
import time
import threading
import urllib.request  # noqa: F401  hsemotion_onnx forgets to import this
import uuid
from pathlib import Path


# --- Self-launch when invoked as `python app.py` -----------------------------
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
# -----------------------------------------------------------------------------

import av
import cv2
import numpy as np
import streamlit as st
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from insightface.app import FaceAnalysis
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

KNOWN_DIR = Path(__file__).parent / "known_faces"
KNOWN_DIR.mkdir(exist_ok=True)
THRESHOLD = 0.42         # cosine sim required to call it a known match
SESSION_DEDUP = 0.55     # cosine sim above which two detections are "same person" in this session
DETECT_EVERY = 2         # run the detector every Nth frame (drawing happens every frame)

# --- Scene-suspicion scoring -------------------------------------------------
NEGATIVE_EMOTIONS = {"Anger", "Fear", "Disgust"}
SUSPICION_THRESHOLD = 0.55      # combined score above this → SUSPICIOUS
MOTION_NORMALIZER = 4.0         # avg flow magnitude that maps to motion=1.0
MOTION_WINDOW = 12              # rolling window of recent motion samples
SCENE_HISTORY = 6               # rolling window of recent suspicion labels
W_MOTION, W_EMOTION, W_CROWD = 0.55, 0.35, 0.10


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


@st.cache_resource(show_spinner="Loading emotion recognizer…")
def get_fer() -> HSEmotionRecognizer:
    """Load HSEmotion (AffectNet, 8 classes). Pre-downloads the ONNX file
    ourselves because the upstream package's auto-downloader has a buggy
    urllib import."""
    cache_dir = Path.home() / ".hsemotion"
    cache_dir.mkdir(exist_ok=True)
    model_name = "enet_b0_8_best_afew"
    fpath = cache_dir / f"{model_name}.onnx"
    if not fpath.exists():
        url = (
            "https://github.com/HSE-asavchenko/face-emotion-recognition"
            f"/raw/main/models/affectnet_emotions/onnx/{model_name}.onnx"
        )
        urllib.request.urlretrieve(url, fpath)
    return HSEmotionRecognizer(model_name=model_name)


def score_suspicion(motion: float, emotions: list[str | None],
                    num_faces: int) -> tuple[str, float, dict]:
    """Combine motion / emotion / crowd signals into a single 0..1 score
    and a NORMAL/SUSPICIOUS label. Returns (label, score, components)."""
    motion_part = min(1.0, max(0.0, motion / MOTION_NORMALIZER))
    valid = [e for e in emotions if e]
    emotion_part = (
        sum(1 for e in valid if e in NEGATIVE_EMOTIONS) / len(valid)
        if valid else 0.0
    )
    crowd_part = min(1.0, num_faces / 4.0)
    score = (W_MOTION * motion_part
             + W_EMOTION * emotion_part
             + W_CROWD * crowd_part)
    label = "SUSPICIOUS" if score >= SUSPICION_THRESHOLD else "NORMAL"
    return label, score, {
        "motion": motion_part,
        "negative_emotion": emotion_part,
        "crowd": crowd_part,
    }


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\- ]", "", name).strip().replace(" ", "_")


def list_people() -> list[str]:
    return sorted(p.name for p in KNOWN_DIR.iterdir() if p.is_dir())


def load_db() -> dict[str, np.ndarray]:
    """Average embedding per known person, L2-normalized."""
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


# --- Live video processor ---------------------------------------------------
class FaceProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model = get_model()
        self.fer = get_fer()
        self.db = load_db()
        self.last_results: list[dict] = []     # detections from last analyzed frame
        self._frame = 0

        # Scene-suspicion state
        self._prev_gray: np.ndarray | None = None
        self._motion_buf: collections.deque[float] = collections.deque(
            maxlen=MOTION_WINDOW
        )
        self._scene_buf: collections.deque[str] = collections.deque(
            maxlen=SCENE_HISTORY
        )
        self.scene: dict = {
            "label": "NORMAL", "score": 0.0,
            "motion": 0.0, "neg_emotion": 0.0, "crowd": 0.0,
        }

    def reload_db(self) -> None:
        self.db = load_db()

    # --- helpers --------------------------------------------------------
    def _emotion_for(self, crop_bgr: np.ndarray) -> str | None:
        if crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < 24:
            return None
        try:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            label, _ = self.fer.predict_emotions(rgb, logits=False)
            return label
        except Exception:
            return None

    def _update_motion(self, img: np.ndarray) -> None:
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (160, 120))
        if self._prev_gray is not None:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    self._prev_gray, small, None,
                    0.5, 2, 15, 2, 5, 1.1, 0,
                )
                mag = float(np.mean(np.linalg.norm(flow, axis=2)))
                self._motion_buf.append(mag)
            except cv2.error:
                pass
        self._prev_gray = small

    def _draw_banner(self, img: np.ndarray) -> None:
        s = self.scene
        color = (0, 0, 220) if s["label"] == "SUSPICIOUS" else (0, 170, 0)
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 44), color, -1)
        cv2.putText(
            img, f"{s['label']}  score={s['score']:.2f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
        )
        sub = (f"motion {s['motion']:.2f}  neg-emo {s['neg_emotion']:.2f}  "
               f"crowd {s['crowd']:.2f}")
        cv2.putText(img, sub, (max(10, w - 360), 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- per-frame entry point -----------------------------------------
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        self._frame += 1

        # Optical flow runs every frame so motion stays responsive
        self._update_motion(img)

        if self._frame % DETECT_EVERY == 0:
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
                emotion = self._emotion_for(crop)
                results.append({
                    "bbox": (x1, y1, x2, y2),
                    "name": name,
                    "sim": sim,
                    "embedding": emb,
                    "crop": crop,
                    "emotion": emotion,
                })

            # Recompute the scene score on each analyzed frame, smoothed
            # via a short rolling history of recent labels.
            avg_motion = (float(np.mean(self._motion_buf))
                          if self._motion_buf else 0.0)
            label, score, parts = score_suspicion(
                avg_motion, [r["emotion"] for r in results], len(results),
            )
            self._scene_buf.append(label)
            # require a majority of recent samples to flip the banner
            sus_votes = sum(1 for x in self._scene_buf if x == "SUSPICIOUS")
            smoothed = ("SUSPICIOUS"
                        if sus_votes > len(self._scene_buf) / 2
                        else "NORMAL")
            with self.lock:
                self.last_results = results
                self.scene = {
                    "label": smoothed,
                    "score": score,
                    "motion": parts["motion"],
                    "neg_emotion": parts["negative_emotion"],
                    "crowd": parts["crowd"],
                }

        # Draw boxes + emotion every frame so overlays look continuous
        with self.lock:
            results = self.last_results
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            color = (0, 200, 0) if r["name"] else (0, 0, 220)
            who = r["name"] if r["name"] else "Unknown"
            emo = f"  {r['emotion']}" if r.get("emotion") else ""
            label = f"{who}  {r['sim']:.2f}{emo}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        self._draw_banner(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- UI ----------------------------------------------------------------------
st.set_page_config(page_title="Live Face Identifier", layout="wide")
st.title("Live Face Identifier")
st.caption("Click START. Faces are detected automatically — green = known, red = unknown.")

# Warm both models so the first webcam frame isn't a stall
get_model()
get_fer()

if "session_faces" not in st.session_state:
    st.session_state.session_faces = {}  # fid -> {name, embedding, crop, first_seen, last_seen, count}

video_col, side_col = st.columns([3, 1], gap="large")

with video_col:
    # Live scene-suspicion banner that mirrors what's burned onto the video.
    banner = st.empty()

    ctx = webrtc_streamer(
        key="face-id",
        video_processor_factory=FaceProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Render the banner from the worker's latest scene state
    scene = (ctx.video_processor.scene
             if ctx.video_processor is not None else None)
    if scene:
        if scene["label"] == "SUSPICIOUS":
            banner.error(
                f"🚨 **SUSPICIOUS** — score {scene['score']:.2f}  "
                f"(motion {scene['motion']:.2f} · "
                f"negative emotions {scene['neg_emotion']:.2f} · "
                f"crowd {scene['crowd']:.2f})"
            )
        else:
            banner.success(
                f"✅ **NORMAL** — score {scene['score']:.2f}  "
                f"(motion {scene['motion']:.2f} · "
                f"negative emotions {scene['neg_emotion']:.2f} · "
                f"crowd {scene['crowd']:.2f})"
            )
    else:
        banner.info("Scene status will appear here once the camera is running.")

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
    if st.button("↻ Reload known faces", width="stretch"):
        if ctx.video_processor:
            ctx.video_processor.reload_db()
        st.toast("Reloaded known faces")

    if st.button("🗑 Reset session", width="stretch", type="primary"):
        st.session_state.session_faces = {}
        st.toast("Session cleared")
        st.rerun()

# Auto-rerun while the camera is playing so the table stays fresh.
# Pause auto-refresh while the user is labeling someone (otherwise the form
# would re-render and the user could lose what they typed).
labeling_in_progress = any(
    k.startswith("labeling_") and v for k, v in st.session_state.items()
)
if ctx.state.playing and not labeling_in_progress:
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
            # Refresh thumbnail occasionally
            if ex["count"] % 30 == 0:
                ex["crop"] = r["crop"]
        else:
            fid = uuid.uuid4().hex[:6]
            st.session_state.session_faces[fid] = {
                "name": r["name"],
                "embedding": r["embedding"].copy(),
                "crop": r["crop"],
                "first_seen": now,
                "last_seen": now,
                "count": 1,
            }

# --- Bottom summary table ---------------------------------------------------
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
