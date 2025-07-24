import torch
import torch.nn.functional as F
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortMatcher:
    """
    Wrapper for DeepSort tracker.
    Parameters:
        max_age (int): Max frames to keep lost tracks.
        n_init (int): Frames before track is confirmed.
        max_cosine_distance (float): Appearance matching threshold.
        nms_max_overlap (float): NMS overlap threshold.
        nn_budget (int or None): Max number of embeddings to keep per track.
        half (bool): Use half precision.
        bgr (bool): Input images are BGR.
        embedder_gpu (bool): Use GPU for embedding.
    """
    def __init__(self, max_age, n_init, max_cosine_distance, nms_max_overlap, nn_budget, half, bgr, embedder_gpu):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=nms_max_overlap,
            embedder=None,
            half=half,
            bgr=bgr,
            embedder_gpu=embedder_gpu,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            override_track_class=None
        )

    def update_tracks(self, boxes, embeddings, frame):
        if len(boxes) == 0 or len(embeddings) == 0 or len(boxes) != len(embeddings):
            print(f"Skipping frame: boxes={len(boxes)}, embeddings={len(embeddings)}")
            return []
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], 1.0, 'person'))
        tracks = self.tracker.update_tracks(detections, embeds=embeddings, frame=frame)
        return tracks