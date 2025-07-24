# fusionreid_tracker/main.py

import cv2
import torch
import numpy as np
import yaml
from models.fusion_reid import FusionReID
from models.cnn_backbone import build_resnet50
from models.transformer_backbone import build_vit_b16
from detectors.yolo_detector import YOLODetector
from reid.matcher import DeepSortMatcher
from utils.preprocessing import preprocess_crop
from utils.visualization import draw_boxes
from deep_sort_realtime.deepsort_tracker import DeepSort

VideoWriter_fourcc = cv2.VideoWriter_fourcc  # type: ignore[attr-defined]

# --- Load Config ---
with open('configs/config.yaml', 'r') as f:
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
# --- Deep SORT config ---
deep_sort_cfg = config.get('deep_sort', {})
matcher = DeepSortMatcher(
    max_age=deep_sort_cfg.get('max_age', 30),
    n_init=deep_sort_cfg.get('n_init', 3),
    max_cosine_distance=deep_sort_cfg.get('max_cosine_distance', 0.2),
    nms_max_overlap=deep_sort_cfg.get('nms_max_overlap', 1.0),
    nn_budget=deep_sort_cfg.get('nn_budget', None),
    half=True if device == 'cuda' else False,
    bgr=True,
    embedder_gpu=(device == 'cuda')
)

# --- Video Input ---
input_video = config.get('video', {}).get('input', 'input.mp4')
output_video = config.get('video', {}).get('output', 'output.mp4')
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open {input_video}")
writer = None

matching_threshold = config.get('matching', {}).get('threshold', 0.4)

frame_idx = 0
tracker = DeepSort(
    max_age=90,  # frames to keep 'lost' tracks
    n_init=3,    # frames before track is confirmed
    nms_max_overlap=1.0,
    embedder=None,  # We'll provide our own embeddings
    half=True if device == 'cuda' else False,
    bgr=True,
    embedder_gpu=device == 'cuda',
    max_cosine_distance=0.2,  # can tune this
    nn_budget=None,
    override_track_class=None
)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
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
    inputs = torch.stack(crops).to(device)
    print(f"inputs shape: {inputs.shape}")
    with torch.no_grad():
        embeddings = model(inputs).cpu().numpy()
        print(f"embeddings shape: {embeddings.shape}")
    # Deep SORT tracking
    tracks = matcher.update_tracks(boxes, embeddings, frame)
    # Draw output
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    if writer is None:
        fourcc = VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, 30, (frame.shape[1], frame.shape[0]))
    writer.write(frame)

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
