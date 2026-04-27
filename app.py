"""
app.py
------
YOLOv8 Object Detector — HuggingFace Spaces Edition
====================================================
Modes:
  1. Image Upload  — detect objects in a single uploaded image.
  2. Video Upload  — process an entire video and download annotated output.

WebRTC webcam is not supported on HuggingFace Spaces.
Run locally for live webcam detection.

Run locally:
    streamlit run app.py
"""

import io
import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from models.loader import MODEL_OPTIONS, load_model
from utils.detection import run_detection, overlay_stats, pil_to_bgr, bgr_to_rgb
from utils.video import (
    iter_video_frames,
    get_video_metadata,
    save_annotated_video,
    make_temp_video_path,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLOv8 Object Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background: #080d1a; color: #e2e8f0; }

  section[data-testid="stSidebar"] {
    background: #0c1120 !important;
    border-right: 1px solid #1a2540;
  }

  /* ── Hero banner ── */
  .hero {
    background: linear-gradient(135deg, #0c1120 0%, #111827 60%, #0a0f1e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -5%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 65%);
    pointer-events: none;
  }
  .hero::after {
    content: '';
    position: absolute;
    bottom: -40%;
    left: 20%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(129,140,248,0.05) 0%, transparent 65%);
    pointer-events: none;
  }
  .hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem;
    line-height: 1.2;
  }
  .hero p { color: #94a3b8; margin: 0; font-size: 0.95rem; }
  .hero .badge {
    display: inline-block;
    background: #1e3a5f;
    color: #38bdf8;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    margin-right: 0.4rem;
    margin-top: 0.7rem;
  }

  /* ── Metric cards ── */
  .metric-row { display: flex; gap: 0.9rem; margin: 1.2rem 0; flex-wrap: wrap; }
  .metric-card {
    background: #0c1120;
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 0.85rem 1.2rem;
    flex: 1;
    min-width: 110px;
  }
  .metric-card .label {
    font-size: 0.67rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 700;
    margin-bottom: 0.3rem;
  }
  .metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    color: #38bdf8;
    line-height: 1.15;
  }

  /* ── Detection table ── */
  .det-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; margin-top: 0.7rem; }
  .det-table th {
    background: #111827;
    color: #64748b;
    padding: 0.5rem 0.8rem;
    text-align: left;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
  }
  .det-table td { padding: 0.45rem 0.8rem; border-bottom: 1px solid #1a2540; color: #cbd5e1; }
  .det-table tr:hover td { background: #1a254015; }
  .conf-bar { height: 5px; border-radius: 3px; background: linear-gradient(90deg, #38bdf8, #818cf8); margin-top: 3px; }

  /* ── Section title ── */
  .section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #475569;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
    margin: 1.4rem 0 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1a2540;
  }

  /* ── Info box ── */
  .info-box {
    background: #0c1829;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #38bdf8;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #94a3b8;
    margin: 0.8rem 0;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.6rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
  .stButton > button:active { transform: translateY(0px) !important; }

  /* ── Download button ── */
  .stDownloadButton > button {
    background: #0f1f35 !important;
    color: #38bdf8 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
  }

  /* ── Widgets ── */
  div[data-baseweb="select"] > div { background: #0c1120 !important; border-color: #1a2540 !important; }
  .stTabs [data-baseweb="tab-list"] { background: #0c1120; border-radius: 8px; border: 1px solid #1a2540; }
  .stTabs [data-baseweb="tab"] { color: #64748b; }
  .stTabs [aria-selected="true"] { color: #38bdf8 !important; }
  .stImage img { border-radius: 10px; border: 1px solid #1a2540; }
  div[data-testid="stFileUploadDropzone"] {
    background: #0c1120 !important;
    border: 2px dashed #1a2540 !important;
    border-radius: 10px !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
if "model_name" not in st.session_state:
    st.session_state.model_name = None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")

    selected_label = st.selectbox(
        "Model variant",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
        help="Nano is fastest; XLarge is most accurate.",
    )
    weights_file = MODEL_OPTIONS[selected_label]

    conf_thresh = st.slider(
        "Confidence threshold", 0.10, 0.95, 0.40, 0.05,
        help="Detections below this score are hidden.",
    )
    iou_thresh = st.slider(
        "IoU (NMS) threshold", 0.10, 0.95, 0.45, 0.05,
        help="Controls duplicate-box suppression.",
    )

    st.markdown("---")
    load_btn = st.button("🔄 Load / Reload Model", use_container_width=True)
    if load_btn or st.session_state.model is None:
        with st.spinner(f"Loading {selected_label}…"):
            try:
                st.session_state.model = load_model(weights_file)
                st.session_state.model_name = selected_label
                st.success(f"✅ Ready — {selected_label}")
            except Exception as exc:
                st.error(f"❌ Model load failed:\n{exc}")
                st.stop()

    st.markdown("---")
    st.markdown("""
    ### 📦 Model sizes
    | Model | Size |
    |---|---|
    | Nano | ~6 MB |
    | Small | ~22 MB |
    | Medium | ~52 MB |
    | Large | ~87 MB |
    | XLarge | ~136 MB |

    Weights are auto-downloaded on first use.
    """)

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem;color:#334155'>Built with Ultralytics YOLOv8 · OpenCV · Streamlit<br>"
        "Deployed on 🤗 HuggingFace Spaces</p>",
        unsafe_allow_html=True,
    )

model = st.session_state.model


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎯 YOLOv8 Object Detector</h1>
  <p>Real-time computer vision — 80 COCO object classes detected instantly</p>
  <div>
    <span class="badge">YOLOv8</span>
    <span class="badge">OpenCV</span>
    <span class="badge">80 Classes</span>
    <span class="badge">🤗 HuggingFace Spaces</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_image, tab_video, tab_info = st.tabs([
    "🖼️  Image Detection",
    "🎬  Video Detection",
    "ℹ️  About & Classes",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab_image:
    st.markdown('<div class="section-title">Upload an image</div>', unsafe_allow_html=True)

    uploaded_image = st.file_uploader(
        "Supported: JPG, JPEG, PNG, BMP, WEBP",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_uploader",
    )

    if uploaded_image and model:
        pil_img = Image.open(uploaded_image)
        bgr_frame = pil_to_bgr(pil_img)

        with st.spinner("Running YOLOv8 inference…"):
            annotated, detections, infer_ms = run_detection(
                model, bgr_frame, conf_thresh, iou_thresh
            )

        # Metrics
        n_det = len(detections)
        class_counts: dict[str, int] = {}
        for d in detections:
            class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1
        top_class = max(class_counts, key=class_counts.get) if class_counts else "—"
        avg_conf = sum(d["confidence"] for d in detections) / n_det if n_det else 0

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="label">Objects detected</div>
            <div class="value">{n_det}</div>
          </div>
          <div class="metric-card">
            <div class="label">Inference time</div>
            <div class="value">{infer_ms:.0f}<span style="font-size:0.9rem;color:#64748b"> ms</span></div>
          </div>
          <div class="metric-card">
            <div class="label">Avg confidence</div>
            <div class="value">{avg_conf:.0%}</div>
          </div>
          <div class="metric-card">
            <div class="label">Top class</div>
            <div class="value" style="font-size:1rem;color:#34d399">{top_class}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Original vs annotated
        col_orig, col_ann = st.columns(2)
        with col_orig:
            st.markdown("**Original**")
            st.image(pil_img, use_container_width=True)
        with col_ann:
            st.markdown("**Detected**")
            rgb_ann = bgr_to_rgb(annotated)
            st.image(rgb_ann, use_container_width=True)

        # Download button
        buf = io.BytesIO()
        Image.fromarray(rgb_ann).save(buf, format="JPEG", quality=92)
        st.download_button(
            "⬇️ Download annotated image",
            data=buf.getvalue(),
            file_name="yolov8_detected.jpg",
            mime="image/jpeg",
        )

        # Detection table
        if detections:
            st.markdown('<div class="section-title">Detection Results</div>', unsafe_allow_html=True)
            rows = "".join(
                f"""<tr>
                  <td>{i+1}</td>
                  <td><b>{d['class_name']}</b></td>
                  <td>
                    {d['confidence']:.1%}
                    <div class="conf-bar" style="width:{d['confidence']*100:.0f}%"></div>
                  </td>
                  <td style="font-size:0.76rem;color:#475569">
                    ({d['bbox'][0]}, {d['bbox'][1]}) → ({d['bbox'][2]}, {d['bbox'][3]})
                  </td>
                </tr>"""
                for i, d in enumerate(sorted(detections, key=lambda x: -x["confidence"]))
            )
            st.markdown(f"""
            <table class="det-table">
              <thead><tr><th>#</th><th>Class</th><th>Confidence</th><th>Bounding Box (px)</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="info-box">⚠️ No objects detected above the confidence threshold. '
                'Try lowering the slider in the sidebar.</div>',
                unsafe_allow_html=True,
            )

    elif not model:
        st.markdown(
            '<div class="info-box">⚠️ Please load a model using the sidebar.</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO
# ══════════════════════════════════════════════════════════════════════════════
with tab_video:
    st.markdown('<div class="section-title">Upload a video</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">💡 Tip: Use a short clip (< 30 sec) for faster results on HuggingFace free tier CPU.</div>',
        unsafe_allow_html=True,
    )

    uploaded_video = st.file_uploader(
        "Supported: MP4, MOV, AVI, MKV, WEBM",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="vid_uploader",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        sample_every = st.number_input(
            "Process every N-th frame",
            min_value=1, max_value=30, value=2,
            help="Set to 2–5 for faster processing on CPU.",
        )
    with col_b:
        max_frames = st.number_input(
            "Max frames (0 = all)",
            min_value=0, max_value=3000, value=150,
            help="Limit frames to avoid timeout on free tier.",
        )

    if uploaded_video and model:
        with tempfile.NamedTemporaryFile(
            suffix=Path(uploaded_video.name).suffix, delete=False
        ) as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = tmp.name

        meta = get_video_metadata(tmp_path)
        source_fps = meta.get("fps", 25.0) or 25.0
        total_frames = meta.get("frame_count", 0) or 0

        st.markdown(f"""
        <div class="info-box">
          📹 <b>Video info:</b> {meta.get('width')}×{meta.get('height')} px ·
          {source_fps:.1f} FPS · {total_frames} frames
        </div>
        """, unsafe_allow_html=True)

        if st.button("▶️ Run Detection on Video"):
            annotated_frames: list[np.ndarray] = []
            total_detections = 0
            frame_idx = 0
            limit = int(max_frames) if max_frames > 0 else None
            vis_total = min(limit or total_frames, total_frames) or 1

            progress_bar = st.progress(0.0, text="Starting…")
            status_text = st.empty()
            preview_slot = st.empty()

            try:
                for raw_frame in iter_video_frames(tmp_path):
                    if limit and frame_idx >= limit:
                        break

                    if frame_idx % int(sample_every) == 0:
                        ann, dets, infer_ms = run_detection(
                            model, raw_frame, conf_thresh, iou_thresh
                        )
                        ann = overlay_stats(ann, None, infer_ms, len(dets))
                        annotated_frames.append(ann)
                        total_detections += len(dets)

                        if len(annotated_frames) % 15 == 0:
                            preview_slot.image(
                                bgr_to_rgb(ann),
                                caption=f"Frame {frame_idx} — {len(dets)} objects",
                                use_container_width=True,
                            )
                    else:
                        # Keep original frame to maintain video timing
                        annotated_frames.append(raw_frame)

                    frame_idx += 1
                    pct = min(frame_idx / vis_total, 1.0)
                    progress_bar.progress(pct, text=f"Frame {frame_idx} / {vis_total}")
                    status_text.markdown(
                        f"Processing… **{frame_idx}** frames done · "
                        f"**{total_detections}** detections so far"
                    )

            except Exception as exc:
                st.error(f"Video processing error: {exc}")
                logger.exception(exc)
            else:
                progress_bar.progress(1.0, text="Done!")
                status_text.success(
                    f"✅ Processed **{frame_idx}** frames · "
                    f"**{total_detections}** total detections"
                )

                out_path = make_temp_video_path(".mp4")
                save_annotated_video(annotated_frames, out_path, fps=source_fps)

                with open(out_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download annotated video",
                        data=f.read(),
                        file_name="yolov8_annotated.mp4",
                        mime="video/mp4",
                    )

    elif not model:
        st.markdown(
            '<div class="info-box">⚠️ Please load a model using the sidebar.</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — INFO
# ══════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.markdown('<div class="section-title">About this app</div>', unsafe_allow_html=True)
    st.markdown("""
    This app runs **YOLOv8** (You Only Look Once v8) from [Ultralytics](https://ultralytics.com),
    a state-of-the-art real-time object detection model trained on the **COCO dataset**.

    | Component | Technology |
    |---|---|
    | Detection model | Ultralytics YOLOv8 |
    | Image processing | OpenCV + Pillow |
    | UI framework | Streamlit |
    | Deployment | 🤗 HuggingFace Spaces |

    ---
    """)

    st.markdown('<div class="section-title">All 80 detectable COCO classes</div>', unsafe_allow_html=True)

    coco_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]

    cols = st.columns(5)
    for i, cls in enumerate(coco_classes):
        cols[i % 5].markdown(f"• {cls}")

    st.markdown("---")
    st.markdown(
        '<div class="info-box">💻 <b>Running locally?</b> Clone the repo and run '
        '<code>streamlit run app.py</code> for webcam live detection support.</div>',
        unsafe_allow_html=True,
    )


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#334155;font-size:0.78rem'>"
    "YOLOv8 Object Detector · Ultralytics · OpenCV · Streamlit · "
    "🤗 HuggingFace Spaces</p>",
    unsafe_allow_html=True,
)
