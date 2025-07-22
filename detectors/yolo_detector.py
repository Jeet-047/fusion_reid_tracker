from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, image):
        results = self.model(image)[0]
        # Only keep boxes for persons (class 0) and above confidence threshold
        return [
            box.xyxy[0].cpu().numpy().astype(int)
            for box in results.boxes
            if int(box.cls[0]) == 0 and float(box.conf[0]) >= self.conf_threshold
        ]