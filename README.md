# 🚀 FusionReID Tracker

A robust, modular pipeline for person re-identification and multi-object tracking using feature fusion of CNN (ResNet50) and Transformer (ViT) backbones, with YOLOv8 for detection.

---

## 📦 Features
- 🧑‍🤝‍🧑 **Person Detection** (YOLOv8, only class 0)
- 🔗 **Feature Fusion** (CNN + ViT)
- 🆔 **Re-ID Matching** (Cosine similarity, memory bank)
- 🏷️ **ID Assignment & Tracking**
- 🎥 **Video Input/Output**
- ⚡ **CUDA/CPU Support**
- 🛠️ **Configurable via YAML**

---

## 🗺️ Workflow

```mermaid
graph TD;
    A[📹 Read Video Frame] --> B[🕵️ Detect Persons (YOLOv8)]
    B --> C[✂️ Crop & Preprocess]
    C --> D[🔬 Extract Embeddings (FusionReID)]
    D --> E[🔎 Match to Memory (Cosine)]
    E --> F[🆔 Assign/Update IDs]
    F --> G[🖼️ Draw Boxes & IDs]
    G --> H[💾 Write to Output Video]
    H --> A
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
3. **Download YOLOv8 weights:**
   - Place `yolov8n.pt` in the project root (or update the config).
4. **Prepare your input video:**
   - Place your video as `input.mp4` (or update the config).

---

## 📝 Configuration

All settings are in `configs/fusion_reid.yaml`:
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

matching:
  threshold: 0.4
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
- **Matching:** Embeddings are compared to a memory bank using cosine similarity. If below the matching threshold, the ID is matched; otherwise, a new ID is assigned.
- **Tracking:** IDs are updated and drawn on the output frame.

---

## 🛠️ Troubleshooting
- **CUDA not used?**
  - Make sure you installed the CUDA-enabled PyTorch and have a compatible GPU.
- **Shape errors?**
  - Ensure your input crops are resized to `(224, 224)`.
- **Push to GitHub fails?**
  - Run `git pull --rebase` before pushing, or use `git push --force` if you want to overwrite remote changes.

---

## 🙋‍♂️ Questions?
Open an issue or discussion on GitHub! 