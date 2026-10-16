"""YOLO detector for cars, motorcycles, buses, and trucks."""
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

class VehicleDetector:
    def __init__(self, model_name="yolo11n.pt", confidence=0.35, image_size=640):
        # Ultralytics downloads model weights on first use.
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, frame):
        results = self.model.predict(
            source=frame, conf=self.confidence, imgsz=self.image_size, verbose=False
        )
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                if class_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                detections.append({
                    "class_id": class_id,
                    "class_name": VEHICLE_CLASSES[class_id],
                    "confidence": float(box.conf[0].item()),
                    "bbox": (x1, y1, x2, y2),
                })
        return detections
