# Traffic Vehicle Detection, Tracking & Counting System

YOLO + OpenCV program to detect cars, motorcycles, buses, and trucks; assign track IDs; estimate line-crossing counts; and export annotated video and performance data.

## Features
- Pretrained YOLO model (`yolo11n.pt`, downloaded on first run)
- Vehicle classes: car, motorcycle, bus, truck
- Lightweight IoU/centroid tracking and a virtual counting line
- Dynamic on-video class totals, labels, confidence, and track IDs
- Gaussian blur + adaptive threshold diagnostic (`--segmentation-view`)
- MP4 output, predictions CSV, frame metrics CSV, and JSON summary
- Optional labelled-ground-truth evaluation for precision, recall, F1, and IoU

## Setup (Windows PowerShell)
Run from the project root:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
If activation is blocked:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Add a traffic video
Copy a traffic video you own or are licensed to use to `data\traffic.mp4`. The project does not bundle third-party video.

## Run
Process all frames:
```powershell
python src/main.py --source data/traffic.mp4
```
Show live preview (press Q to stop):
```powershell
python src/main.py --source data/traffic.mp4 --display
```
Show adaptive threshold diagnostic:
```powershell
python src/main.py --source data/traffic.mp4 --segmentation-view
```
Test only first 300 frames:
```powershell
python src/main.py --source data/traffic.mp4 --max-frames 300
```
Use webcam:
```powershell
python src/main.py --source 0 --display
```
Adjust confidence and line:
```powershell
python src/main.py --source data/traffic.mp4 --confidence 0.40 --line-position 0.55
```

First run needs internet to download the checkpoint. If `yolo11n.pt` is not recognized, update Ultralytics with `python -m pip install -U ultralytics`.

## Outputs
- `outputs/annotated_traffic.mp4`: boxes, labels, IDs, counting line, dynamic totals
- `outputs/predictions.csv`: frame-level boxes and confidence
- `outputs/frame_metrics.csv`: detections and processing time per frame
- `outputs/summary.json`: runtime stats and line-crossing totals

## Tests
```powershell
python -m unittest discover -s tests -v
```

## Evaluate accuracy with labelled ground truth
Create `data/ground_truth.csv` with columns:
```csv
frame,class_name,x1,y1,x2,y2
1,car,100,80,180,150
1,bus,260,60,390,190
```
Annotate the same frames the program processes, then:
```powershell
python src/evaluate.py --predictions outputs/predictions.csv --ground-truth data/ground_truth.csv --iou 0.5
```
Do not report accuracy values without ground-truth annotations.

## Pipeline
1. Read frame with OpenCV.
2. Run YOLO on the original color frame for semantic vehicle detection.
3. Match detections with a class-aware IoU/centroid tracker.
4. Count a tracked vehicle the first time its center crosses the horizontal line.
5. Render boxes, track IDs, confidence, line, and live per-class totals.
6. Export video, CSVs, and performance JSON.
7. Optional diagnostic applies Gaussian blur and adaptive thresholding. This mask is not used as YOLO input because it can remove features needed by the model.

## Limitations
The tracker is an educational baseline, not as robust as ByteTrack/BoT-SORT. Occlusion and missed detections may change IDs and counts. Performance depends on the camera, light, resolution, model, and hardware. CPU-only inference may be slow. Validate on representative labelled traffic footage before real operational use.
