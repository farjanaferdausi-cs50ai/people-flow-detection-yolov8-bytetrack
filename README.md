# People Flow Detection using Object Tracking & Heatmap Visualization

![Python](https://img.shields.io/badge/Python-3.x-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-brightgreen)
![ByteTrack](https://img.shields.io/badge/Tracker-ByteTrack-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-opencv--python--headless-red)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow)

## Overview

I built this project to detect, track, and count people moving through a video scene in real time, and to visualize *how* they move through the space using motion heatmaps. It's an end-to-end computer vision pipeline that combines person detection, multi-object tracking, bidirectional line-crossing analytics, and heatmap generation — submitted as my Module 16 assignment for the **Ostad AI/ML Engineering Program**.

The problem I set out to solve: given a top-down video feed of people walking through an open area, how do I accurately count how many people enter and exit through two defined zones — without double-counting a single person — while also surfacing which paths people take most often?

## Key Features

- **Person detection** with YOLOv8n (COCO-pretrained), filtered to a single class (`person`) for speed
- **Multi-object tracking** with ByteTrack, giving every person a persistent ID across frames
- **Bidirectional IN/OUT counting** across two independently configurable virtual lines
- **Double-count protection** using per-ID crossing-state tracking, so a person oscillating near a line is never counted twice
- **Motion heatmap generation** — a standalone JET-colormap heatmap and an INFERNO-colormap heatmap overlaid on the real scene
- **Fully annotated output video** with bounding boxes, track IDs, live IN/OUT counters, and the two virtual lines drawn in
- **Fully configurable pipeline** — every parameter (video source, line positions, confidence threshold, frame skip) lives in a single `CONFIG` dictionary

## Tech Stack

| Component | Choice |
|---|---|
| Language | Python 3 |
| Detection Model | Ultralytics YOLOv8n (`yolov8n.pt`, COCO-pretrained) |
| Tracker | ByteTrack (`bytetrack.yaml`, via Ultralytics' native `model.track()`) |
| Computer Vision | OpenCV (`opencv-python-headless`) |
| Numerical Processing | NumPy |
| Visualization | Matplotlib |
| Environment | Google Colab + Google Drive |

## How It Works — Pipeline

1. **Configuration** — every tunable parameter (video source, IN/OUT line Y-coordinates, confidence threshold, frame skip) is centralized in one `CONFIG` dictionary, so the whole pipeline can be re-pointed at a new video without touching the core logic.
2. **Video Ingestion** — I use `people-walking.mp4`, a public top-down surveillance-style clip from Roboflow's supervision example set (1920×1080 @ 25 FPS), downloaded automatically at runtime.
3. **Detection + Tracking** — each frame is passed through YOLOv8n restricted to class `0` (person) at a confidence threshold of 0.5. The detections are handed to ByteTrack via `model.track(..., persist=True, tracker='bytetrack.yaml')`, which assigns a consistent track ID to each person across frames.
4. **Line-Crossing Counting Logic**
   - **IN line** at Y = 250, **OUT line** at Y = 450 — I determined these coordinates visually using Roboflow's [PolygonZone](https://polygonzone.roboflow.com/) tool.
   - For every tracked ID, I compare the centroid's current Y-coordinate against its Y-coordinate from the previous frame.
   - **IN** is counted the moment a centroid crosses from above Y=250 to at-or-below it.
   - **OUT** is counted the moment a centroid crosses from below Y=450 to at-or-above it.
   - I prevent double-counting with two state sets (`crossed_in`, `crossed_out`). Once an ID is counted for a direction, it can't be re-counted for the same crossing — and if it later performs a genuine crossing in the opposite direction, its state resets correctly instead of getting stuck.
5. **Heatmap Accumulation** — every detected centroid increments a cell in a `float32` accumulator array the size of the frame, building up a raw density map over the full video.
6. **Heatmap Rendering** — from that single accumulator, I render two different visualizations:
   - `final_heatmap.png` — normalized → Gaussian-blurred (31×31 kernel) → re-normalized → JET colormap, shown standalone.
   - `heatmap_overlay.png` — same processing pipeline, INFERNO colormap, blended 70% heatmap / 30% original frame (`cv2.addWeighted`) on top of the last captured frame for spatial context.
7. **Output & Persistence** — the annotated video, both heatmaps, and an auto-generated `README.md` are saved locally, downloaded, and copied to Google Drive (`Module_16_Assignment`).

## Results

Tested on `people-walking.mp4` (1920×1080 @ 25 FPS):

| Metric | Value |
|---|---|
| Frames Processed | 341 |
| Total IN | 10 |
| Total OUT | 8 |
| Unique IDs Tracked | 77 |
| Peak Raw Heatmap Density | 19.0 |
| Tracker | ByteTrack (`bytetrack.yaml`) |

### Sample Output

*(Add your actual output images here after uploading them to the repo — the original clean versions were saved by the notebook to your Google Drive under `Module_16_Assignment` and to your Downloads folder.)*


![Annotated Frame](output_frame_sample.png)
![Standalone Heatmap - JET](final_heatmap.png)
![Heatmap Overlay - INFERNO](heatmap_overlay.png)


## Project Structure

```
├── Module_16_People_Flow_Detection__Farjana_Ferdausi_.ipynb   # Main notebook (Google Colab)
├── output_people_flow.mp4                                     # Annotated output video
├── final_heatmap.png                                          # Standalone motion heatmap (JET)
├── heatmap_overlay.png                                        # Heatmap overlaid on scene (INFERNO)
└── README.md
```

## How to Run

1. Open the notebook in Google Colab.
2. Run all cells sequentially — dependencies (`ultralytics`, `opencv-python-headless`, `numpy`, `matplotlib`) install automatically.
3. The input video downloads automatically from the Roboflow URL — no manual setup required.
4. To use your own footage, edit the `CONFIG` dictionary at the top of the notebook: point `video_url` / `video_name` at your file, and reposition `line_in_y` / `line_out_y` to match your camera's framing (I used PolygonZone to pick mine).
5. Outputs — the annotated video and both heatmap images — are generated in the Colab environment and copied to Google Drive automatically.

## What I'd Improve Next

- Replace fixed pixel-based line coordinates with a one-time interactive calibration step, so the same notebook generalizes to any camera angle without manually re-tuning `line_in_y` / `line_out_y`.
- Add per-zone dwell-time analytics on top of the existing heatmap accumulator.
- Swap YOLOv8n for a larger checkpoint (or a fine-tuned model) in scenarios where detection precision matters more than inference speed.
- Wrap the pipeline in a lightweight Streamlit dashboard so non-technical stakeholders can upload a video and see live counts without touching the notebook.

## 🖊️ Author

**Farjana Ferdausi**
AI/ML Engineering & Data Science, Fellow — Google Cloud Gen AI Academy APAC Edition (Cohort 3) | Agentic AI · RAG · Gemini · ADK · BigQuery MCP · Cloud Run | Former HR Professional (14+ years) at Radisson Blu Dhaka Water Garden, Bangladesh

LinkedIn: [linkedin.com/in/farjana-ferdausi](https://www.linkedin.com/in/farjana-ferdausi/)

Medium: [medium.com/@farjana.rafi1983](https://medium.com/@farjana.rafi1983)
