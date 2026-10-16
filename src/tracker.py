"""Class-aware greedy IoU/centroid tracker for educational use."""
from math import hypot

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(0, ax2-ax1)*max(0, ay2-ay1) + max(0, bx2-bx1)*max(0, by2-by1) - inter
    return inter / union if union else 0.0

class CentroidTracker:
    def __init__(self, max_missed=12, min_iou=0.05, max_distance=90):
        self.max_missed, self.min_iou, self.max_distance = max_missed, min_iou, max_distance
        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def center(box):
        x1, y1, x2, y2 = box
        return ((x1+x2)//2, (y1+y2)//2)

    def update(self, detections):
        unmatched = set(range(len(detections)))
        for track in self.tracks.values():
            track["missed"] += 1
        candidates = []
        for tid, track in self.tracks.items():
            for idx, det in enumerate(detections):
                if track["class_name"] != det["class_name"]:
                    continue
                overlap = box_iou(track["bbox"], det["bbox"])
                a, b = self.center(track["bbox"]), self.center(det["bbox"])
                distance = hypot(a[0]-b[0], a[1]-b[1])
                if overlap >= self.min_iou or distance <= self.max_distance:
                    candidates.append((overlap, -distance, tid, idx))
        candidates.sort(reverse=True)
        used_tracks = set()
        for _, _, tid, idx in candidates:
            if idx not in unmatched or tid in used_tracks:
                continue
            self.tracks[tid].update(bbox=detections[idx]["bbox"], missed=0)
            detections[idx]["track_id"] = tid
            unmatched.remove(idx)
            used_tracks.add(tid)
        for idx in sorted(unmatched):
            det = detections[idx]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"bbox": det["bbox"], "class_name": det["class_name"], "missed": 0}
            det["track_id"] = tid
        self.tracks = {tid: t for tid, t in self.tracks.items() if t["missed"] <= self.max_missed}
        return detections
