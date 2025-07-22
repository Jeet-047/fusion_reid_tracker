# fusionreid_tracker/main.py

import cv2
import torch
import numpy as np
import yaml
from models.fusion_reid import FusionReID
from models.cnn_backbone import build_resnet50
from models.transformer_backbone import build_vit_b16
from detectors.yolo_detector import YOLODetector
from reid.matcher import match_embeddings
from reid.id_manager import IDManager
from utils.preprocessing import preprocess_crop
from utils.visualization import draw_boxes

VideoWriter_fourcc = cv2.VideoWriter_fourcc  # type: ignore[attr-defined]

# --- Load Config ---
with open('configs/fusion_reid.yaml', 'r') as f:
    config = yaml.safe_load(f)

# --- Initialize Modules ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model config
detector_conf = config.get('detector', {}).get('conf_threshold', 0.4)
cnn_backbone = build_resnet50() if config.get('model', {}).get('cnn_backbone', 'resnet50') == 'resnet50' else build_resnet50()
transformer_backbone = build_vit_b16() if config.get('model', {}).get('transformer_backbone', 'vit_b_16') == 'vit_b_16' else build_vit_b16()
dim = config.get('model', {}).get('dim', 768)
num_htm_layers = config.get('model', {}).get('num_htm_layers', 2)

model = FusionReID(
    cnn_backbone=cnn_backbone,
    transformer_backbone=transformer_backbone,
    dim=dim,
    num_htm_layers=num_htm_layers
).to(device)
model.eval()

detector_model_path = config.get('detector', {}).get('model_path', 'yolov8n.pt')
detector = YOLODetector(model_path=detector_model_path, conf_threshold=detector_conf)
id_manager = IDManager()

# --- Video Input ---
input_video = config.get('video', {}).get('input', 'input.mp4')
output_video = config.get('video', {}).get('output', 'output.mp4')
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open {input_video}")
writer = None

matching_threshold = config.get('matching', {}).get('threshold', 0.4)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx+= 1
    # Detect persons
    boxes = detector.detect(frame)
    crops = []
    for box in boxes:
        crop = preprocess_crop(frame, box)
        if not isinstance(crop, torch.Tensor):
            crop = torch.from_numpy(np.array(crop))
        crops.append(crop)

    if len(crops) == 0:
        continue

    # Re-ID Embeddings
    try:
        inputs = torch.stack(crops).to(device)
    except Exception as e:
        print(f"Skipping frame due to crop error: {e}")
        continue
    with torch.no_grad():
        embeddings = model(inputs).cpu()

    # Match IDs
    matched_ids = match_embeddings(embeddings, id_manager, threshold=matching_threshold)
    for i, pid in enumerate(matched_ids):
        if pid == -1:
            pid = id_manager.get_next_id()
        id_manager.update(pid, embeddings[i])

    # Draw output
    frame = draw_boxes(frame, boxes, matched_ids)

    if writer is None:
        fourcc = VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, 30, (frame.shape[1], frame.shape[0]))
    writer.write(frame)

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
