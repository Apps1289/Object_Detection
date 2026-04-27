# 🎯 YOLOv8 Real-Time Object Detector

A production-ready computer vision app — detect objects in images, videos, and live webcam using **YOLOv8 + OpenCV + Streamlit**, deployed on **Render**.

---

## 🚀 Deploy on Render (Step-by-Step)

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USERNAME/yolov8-detector.git
git push -u origin main
```

### 2. Create a Render Web Service

1. Go to **https://render.com** → Sign in → **New → Web Service**
2. Connect your GitHub account and select the `yolov8-detector` repo
3. Fill in the settings:

| Field | Value |
|---|---|
| **Name** | `yolov8-object-detector` |
| **Runtime** | `Python 3` |
| **Region** | Oregon (or nearest) |
| **Branch** | `main` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true` |
| **Plan** | Starter (free) or Standard (recommended for faster inference) |

4. Under **Advanced → Environment Variables**, add:

| Key | Value |
|---|---|
| `YOLO_CONFIG_DIR` | `/tmp/ultralytics` |
| `MPLCONFIGDIR` | `/tmp/matplotlib` |

5. Click **Create Web Service** — Render builds and deploys automatically.

> ⏱️ First deploy takes ~4–6 minutes (installing PyTorch).  
> 🔁 Model weights download on first request, then stay cached in `/tmp`.

---

### Alternative: Use render.yaml (one-click)

If you push `render.yaml` (included), Render auto-configures everything.  
Just click **New → Blueprint** and point to your repo.

---

## ✨ Features

| | |
|---|---|
| 🖼️ **Image Detection** | Upload JPG/PNG/WEBP → bounding boxes + download |
| 🎬 **Video Detection** | Upload MP4/MOV/AVI → frame-by-frame with live preview |
| 📷 **Webcam (Live)** | WebRTC real-time stream with FPS overlay |
| 🔧 **5 Model Sizes** | Nano → XLarge; swap any time |
| 📊 **Confidence Scores** | Per-detection bar + percentage |
| ⚡ **FPS Counter** | Live EMA FPS on webcam stream |

---

## 🗂️ Project Structure

```
yolov8_detector/
├── app.py                  ← Main Streamlit app
├── requirements.txt        ← Python dependencies
├── render.yaml             ← Render one-click deploy config
├── Dockerfile              ← Optional Docker deploy
├── README.md
├── models/
│   ├── __init__.py
│   └── loader.py           ← Model download + /tmp cache logic
└── utils/
    ├── __init__.py
    ├── detection.py        ← Inference, bbox drawing, overlays
    └── video.py            ← Frame iter, FPS tracker, video writer
```

---

## 💻 Run Locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
# Open http://localhost:8501
```

---

## 🧠 Model Sizes (COCO 80 classes)

| Model | Weights | mAP | Best for |
|---|---|---|---|
| YOLOv8n | ~6 MB | 37.3 | Free tier / fast |
| YOLOv8s | ~22 MB | 44.9 | Balanced |
| YOLOv8m | ~52 MB | 50.2 | Better accuracy |
| YOLOv8l | ~87 MB | 52.9 | High accuracy |
| YOLOv8x | ~136 MB | 53.9 | Best accuracy |

> **Render free tier** (512 MB RAM): use **YOLOv8n or YOLOv8s**.  
> **Render Starter** (2 GB RAM): any model works fine.

---

## ⚙️ Render-Specific Notes

- **Filesystem**: Render's filesystem is **ephemeral** — weights are cached in `/tmp` (set via `YOLO_CONFIG_DIR` env var). They re-download on cold start (~5 sec for nano).
- **Sleep on inactivity**: Free tier spins down after 15 min. First request after sleep will be slow (~30 sec).
- **Webcam**: WebRTC runs peer-to-peer — webcam video never touches the Render server.
- **Port**: Render injects `$PORT` automatically; the start command passes it to Streamlit.

---

## 📄 License

MIT
