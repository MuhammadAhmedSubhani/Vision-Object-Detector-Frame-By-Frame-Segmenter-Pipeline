"""YOLO traffic detection, simple tracking, line counting, and export."""
import argparse, csv, json, time
from collections import Counter
from pathlib import Path
import cv2
from vehicle_detector import VehicleDetector
from tracker import CentroidTracker
from metrics import RuntimeMetrics
from vision_utils import make_segmentation_view

CLASSES = ("car", "motorcycle", "bus", "truck")
COLORS = {"car": (60,220,60), "motorcycle": (255,180,30), "bus": (30,200,255), "truck": (220,80,220)}

def source_value(s):
    return int(s) if s.isdigit() else s

def draw_overlay(frame, detections, counts, line_y, frame_no, segmentation=False):
    h, w = frame.shape[:2]
    cv2.line(frame, (0,line_y), (w,line_y), (0,255,255), 2)
    cv2.putText(frame, "COUNTING LINE", (12,max(22,line_y-8)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,255), 2)
    for d in detections:
        x1,y1,x2,y2 = d["bbox"]
        color = COLORS[d["class_name"]]
        label = f'{d["class_name"]} ID:{d["track_id"]} {d["confidence"]:.2f}'
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label, (x1,max(18,y1-7)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
    cv2.rectangle(frame, (8,8), (270,132), (25,25,25), -1)
    cv2.putText(frame, "TRAFFIC VEHICLE COUNTER", (16,29), cv2.FONT_HERSHEY_SIMPLEX, .52, (255,255,255), 2)
    y = 51
    for name in CLASSES:
        cv2.putText(frame, f"{name.title()}: {counts[name]}", (16,y), cv2.FONT_HERSHEY_SIMPLEX, .53, COLORS[name], 2)
        y += 19
    cv2.putText(frame, f"Frame: {frame_no}", (16, y+2), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
    if segmentation:
        mask = cv2.cvtColor(make_segmentation_view(frame), cv2.COLOR_GRAY2BGR)
        sw, sh = max(1,w//4), max(1,h//4)
        mask = cv2.resize(mask, (sw,sh))
        frame[h-sh:h, w-sw:w] = mask
    return frame

def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    ap = argparse.ArgumentParser(description="Detect, track, and count road vehicles in video.")
    ap.add_argument("--source", default="data/traffic.mp4", help="Video path or webcam index (0)")
    ap.add_argument("--model", default="yolo11n.pt", help="Ultralytics model checkpoint")
    ap.add_argument("--confidence", type=float, default=.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--line-position", type=float, default=.60, help="Counting line fraction of image height")
    ap.add_argument("--max-frames", type=int, default=0, help="0 processes all frames")
    ap.add_argument("--display", action="store_true", help="Show live preview; press Q to stop")
    ap.add_argument("--segmentation-view", action="store_true", help="Show Gaussian/adaptive threshold diagnostic")
    ap.add_argument("--output-dir", default="outputs")
    args = ap.parse_args()
    if not 0 <= args.confidence <= 1 or not 0 < args.line_position < 1:
        raise SystemExit("Confidence must be 0-1 and line-position must be between 0 and 1.")
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(source_value(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.source}. Put a video at data/traffic.mp4 or pass --source.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps > 240: fps = 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release(); raise SystemExit("Could not read video dimensions.")
    line_y = int(height * args.line_position)
    video_path = outdir / "annotated_traffic.mp4"
    video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    if not video_writer.isOpened():
        cap.release(); raise SystemExit(f"Cannot write output video: {video_path}")
    detector = VehicleDetector(args.model, args.confidence, args.imgsz)
    tracker = CentroidTracker()
    metrics = RuntimeMetrics()
    counts = Counter({n:0 for n in CLASSES})
    previous_side, counted = {}, set()
    frame_rows, prediction_rows = [], []
    frame_no = total_detections = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if args.max_frames and frame_no >= args.max_frames: break
            frame_no += 1
            start = time.perf_counter()
            detections = tracker.update(detector.detect(frame))
            latency = (time.perf_counter()-start)*1000
            metrics.record(latency)
            total_detections += len(detections)
            for d in detections:
                x1,y1,x2,y2 = d["bbox"]
                cx,cy = (x1+x2)//2, (y1+y2)//2
                tid, name = d["track_id"], d["class_name"]
                side = "above" if cy < line_y else "below"
                if tid in previous_side and previous_side[tid] != side and tid not in counted:
                    counts[name] += 1; counted.add(tid)
                previous_side[tid] = side
                prediction_rows.append({"frame":frame_no,"track_id":tid,"class_id":d["class_id"],
                    "class_name":name,"confidence":round(d["confidence"],5),"x1":x1,"y1":y1,
                    "x2":x2,"y2":y2,"center_x":cx,"center_y":cy})
            video_writer.write(draw_overlay(frame.copy(), detections, counts, line_y, frame_no, args.segmentation_view))
            frame_rows.append({"frame":frame_no,"detections":len(detections),
                "counted_car":counts["car"],"counted_motorcycle":counts["motorcycle"],
                "counted_bus":counts["bus"],"counted_truck":counts["truck"],"processing_ms":round(latency,3)})
            if args.display:
                cv2.imshow("Traffic Vehicle Detection and Counting", draw_overlay(frame.copy(), detections, counts, line_y, frame_no))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break
    finally:
        cap.release(); video_writer.release(); cv2.destroyAllWindows()
    summary = metrics.summary(frame_no, total_detections)
    summary.update({"source":args.source,"model":args.model,"confidence_threshold":args.confidence,
        "vehicle_counts_crossing_line":dict(counts),"unique_track_ids_crossed":len(counted),
        "counting_line_y":line_y,
        "note":"Line-crossing totals are estimates. ID switches, missed detections, and occlusion can affect counts."})
    write_csv(outdir/"frame_metrics.csv", frame_rows)
    write_csv(outdir/"predictions.csv", prediction_rows)
    (outdir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nAnnotated video: {video_path}\nCSV/JSON outputs: {outdir.resolve()}")

if __name__ == "__main__":
    main()
