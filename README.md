# PPE Detection — Construction Site Safety

A desktop application (PyQt6 + OpenCV + YOLOv8) that detects Personal
Protective Equipment in construction-site videos, evaluates per-person
compliance, and stores all results in **JSON files** (no database).

## Features

- **Video Analysis** — load an `.mp4`/`.avi` video, run YOLOv8 frame by frame,
  and watch the annotated stream with **Start / Pause / Stop** controls.
- **AI detection** — person + PPE (helmet, vest, gloves, goggles, mask) using
  your trained YOLOv8 weights.
- **Compliance logic** — each detected person is labelled **Compliant** or
  **Non-compliant** based on the required PPE.
- **Statistics** — live, in-memory totals (persons, violations, compliance %)
  that reset when a new video is loaded.
- **JSON storage** — every processed video appends a record to
  `data/results.json`; reload and browse past results in the GUI.
- **Outputs** — annotated video saved to `data/processed/`, violation
  screenshots to `data/screenshots/`, and an exportable JSON report.

## Project structure

```
project/
├── main.py                  # entry point
├── requirements.txt
├── detector/
│   ├── config.py            # paths, thresholds, CLASS MAPPING, PPE rules
│   ├── ppe_detector.py      # YOLOv8 loading + inference -> normalized detections
│   └── compliance.py        # associate PPE to persons, decide compliance
├── utils/
│   ├── json_storage.py      # append / load / export records (atomic writes)
│   └── visualization.py     # draw boxes + compliance overlay
├── gui/
│   ├── main_window.py       # sidebar + stacked pages, signal wiring
│   ├── video_widget.py      # Video Analysis page
│   ├── statistics_widget.py # Statistics page
│   ├── export_widget.py     # Export Results page
│   └── processing_worker.py # QThread: detection runs off the GUI thread
├── models/                  # put your trained .pt here (ppe_yolov8.pt)
└── data/                    # JSON results, processed videos, screenshots
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires **Python 3.12+**. `ultralytics` pulls in PyTorch; a CUDA GPU is
optional but greatly speeds up inference (set `DEVICE = "0"` in
`detector/config.py`).

## You must supply a trained model

The app does **not** ship weights. Train a YOLOv8 model on your PPE dataset and
place the file at `models/ppe_yolov8.pt` (or pick any `.pt` via *Select Model*).

### Training on the Kaggle dataset (`beyzakucuk/ppe-detection-v1`)

That dataset is built on the *Construction Site Safety* set whose classes are:

```
Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person,
Safety Cone, Safety Vest, machinery, vehicle
```

Train, for example:

```bash
yolo detect train model=yolov8n.pt data=path/to/data.yaml epochs=100 imgsz=640
cp runs/detect/train/weights/best.pt models/ppe_yolov8.pt
```

> **Important — this dataset has no `gloves` or `goggles` class.**
> The project spec asks for helmet/vest/gloves as *required* PPE
> (`REQUIRED_PPE` in `detector/config.py`). To avoid flagging everyone as
> non-compliant for PPE the model physically cannot see, the
> `ComplianceChecker` **automatically drops any required item that the loaded
> model can't detect** and logs which rules were skipped. So with this dataset
> only helmet and vest are enforced. To enforce gloves/goggles, train on a
> dataset that includes those classes (e.g. SH17), then they activate
> automatically.

### Adapting the class names

If your model's class names differ, edit `CLASS_ALIASES` in
`detector/config.py`. Run `python -c "from ultralytics import YOLO;
print(YOLO('models/ppe_yolov8.pt').names)"` to see the exact names, then map
each one to a canonical category (`person`, `helmet`, `vest`, `gloves`,
`goggles`, `mask`, or the negative `no_*` variants). Unmapped clutter classes
(cones, vehicles) should map to `_ignore`.

The dataset's `NO-Hardhat` / `NO-Safety Vest` / `NO-Mask` classes are used as
**direct violation signals** when associated with a person, in addition to the
"PPE simply not detected" rule.

## How it works

1. Choose a video and a model on the **Video Analysis** page, press **Start**.
2. A `QThread` (`ProcessingWorker`) loads the model, then for each frame runs
   detection, associates each PPE box with the person whose box contains it,
   decides compliance, draws the overlay, and emits the frame to the GUI.
3. On completion a record is appended to `data/results.json`, the processed
   video and violation screenshots are saved, and the **Export Results** table
   refreshes. **Statistics** update live and reset on each new video.

## JSON record format

```json
{
  "video_name": "site.mp4",
  "timestamp": "2026-06-03T12:34:56",
  "frames_processed": 300,
  "fps": 25.0,
  "total_persons_detected": 742,
  "max_persons_in_frame": 5,
  "total_violations": 188,
  "compliance_rate": 0.7466,
  "violations": [
    {
      "frame_number": 57,
      "timestamp_sec": 2.28,
      "missing_ppe": ["vest"],
      "bbox": [120, 90, 260, 480],
      "screenshot_path": "data/screenshots/site_frame57.jpg"
    }
  ],
  "processed_video_path": "data/processed/site_processed.mp4"
}
```

## Notes & limitations

- **Person counting:** without object tracking, `total_persons_detected` is the
  *sum of person detections across frames*, not unique individuals
  (`max_persons_in_frame` is the peak in any single frame). For unique counts,
  enable Ultralytics tracking (`model.track(..., persist=True)`) and key the
  stats by track ID.
- **PPE→person association** is a geometric heuristic (center-in-box +
  containment); a pose/keypoint model would improve glove/goggle attribution.
- Processing speed is bound by inference, so playback runs as fast as the model
  allows rather than at the video's native FPS.
```
