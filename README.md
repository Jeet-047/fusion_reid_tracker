# 🚀 FusionReID Tracker

A robust, modular pipeline for person re-identification and multi-object tracking using feature fusion of CNN (ResNet50) and Transformer (ViT) backbones, with YOLOv8 for detection and Deep SORT for robust tracking.

---

## 📦 Features
- 🧑‍🤝‍🧑 **Person Detection** (YOLOv8, only class 0)
- 🔗 **Feature Fusion** (CNN + ViT)
- 🆔 **Re-ID Matching & Tracking** (Deep SORT, Hungarian algorithm, memory averaging)
- 🏷️ **ID Assignment & Tracking**
- 🎥 **Video Input/Output**
- ⚡ **CUDA/CPU Support**
- 🛠️ **Fully Configurable via config.yaml**

---

## 🗺️ Workflow

```
📹 Read Video Frame
   ↓
🕵️ Detect Persons (YOLOv8)
   ↓
✂️ Crop & Preprocess
   ↓
🔬 Extract Embeddings (FusionReID)
   ↓
🔎 Deep SORT: Appearance + Motion Matching (Hungarian algorithm)
   ↓
🆔 Assign/Update IDs
   ↓
🖼️ Draw Boxes & IDs
   ↓
💾 Write to Output Video
   ↓
(repeat for next frame)
```

---

## ⚙️ Setup

1. **Clone the repo:**
   ```sh
   git clone https://github.com/YOUR_USERNAME/fusion_reid_tracker.git
   cd fusion_reid_tracker
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   - This includes `deep_sort_realtime`, `ultralytics`, `torch`, `torchvision`, `opencv-python`, `numpy`, and `pillow`.
3. **Download YOLOv8 weights:**
   - Place `yolov8n.pt` in the project root (or update the config).
4. **Prepare your input video:**
   - Place your video as `input.mp4` (or update the config).

---

## 📝 Configuration

All settings are in `configs/config.yaml`:
```yaml
model:
  cnn_backbone: resnet50
  transformer_backbone: vit_b_16
  dim: 768
  num_htm_layers: 2

detector:
  model_path: yolov8n.pt
  conf_threshold: 0.4

video:
  input: input.mp4
  output: output.mp4

deep_sort:
  max_age: 30
  n_init: 3
  max_cosine_distance: 0.2
  nms_max_overlap: 1.0
  nn_budget: null
```

---

## ▶️ Usage

```sh
python main.py
```
- Output video with tracked IDs will be saved as `output.mp4`.
- All parameters can be changed in the YAML config.

---

## 🧠 How It Works
- **Detection:** YOLOv8 finds all persons in each frame (class 0 only, with confidence threshold).
- **Preprocessing:** Each detected person is cropped, resized, and normalized.
- **Embedding:** Crops are passed through FusionReID (ResNet50 + ViT) to get robust embeddings.
- **Tracking & Re-ID:** Deep SORT uses your embeddings and motion to assign consistent IDs, using the Hungarian algorithm and memory averaging.
- **Visualization:** IDs and boxes are drawn on the output frame.

---

## 🛠️ Troubleshooting
- **CUDA not used?**
  - Make sure you installed the CUDA-enabled PyTorch and have a compatible GPU.
- **Shape errors?**
  - Ensure your input crops are resized to `(224, 224)`.
- **Push to GitHub fails?**
  - Run `git pull --rebase` before pushing, or use `git push --force` if you want to overwrite remote changes.
- **YOLO output always shows `0:`?**
  - This is normal: YOLO prints `0:` for each frame since you process one frame at a time.

---

## 🙋‍♂️ Questions?
Open an issue or discussion on GitHub! 