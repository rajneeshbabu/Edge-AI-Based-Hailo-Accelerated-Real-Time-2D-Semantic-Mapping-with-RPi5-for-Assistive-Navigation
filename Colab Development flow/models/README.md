# Model Files

All compiled and intermediate model artifacts from the training pipeline are stored here. No recompilation is needed to deploy — use `yolov8n_pruned.hef` directly.

---

## Files

| File | Size | Stage | Description |
|---|---|---|---|
| `yolov8n_pruned.hef` | 4.3 MB | ✅ Final | **Hailo Executable Format — deploy this on RPi5 + Hailo-8 AI HAT** |
| `best_opset11.onnx` | 13 MB | Stage 4 | Pruned YOLOv8n ONNX export (opset-11) — input to the Hailo DFC compiler |
| `yolov8n_pruned.har` | 13 MB | Phase B | Hailo Archive — INT8-quantized intermediate produced by `hailo optimize` |
| `best_pruned.pt` | 6.3 MB | Stage 4 | Pruned YOLOv8n PyTorch checkpoint — resume fine-tuning from here |

---

## Pipeline Stage Map

```
best_pruned.pt          ← L1 pruned + fine-tuned PyTorch model (Stage 4)
    ↓  ONNX export (opset-11)
best_opset11.onnx       ← ONNX intermediate (input to Hailo DFC)
    ↓  hailo parser + hailo optimize (INT8, 64 calibration images)
yolov8n_pruned.har      ← Hailo Archive (quantized, before final compile)
    ↓  hailo compile
yolov8n_pruned.hef      ← ✅ Deployed on RPi5 + Hailo-8
```

---

## Deploying the HEF

Transfer `yolov8n_pruned.hef` to the RPi5:

```bash
# Windows PowerShell (replace <RPI_IP>)
scp yolov8n_pruned.hef rpi15@<RPI_IP>:/home/rpi15/edge_nav/models/yolov8n_pruned.hef

# macOS / Linux
scp yolov8n_pruned.hef rpi15@<RPI_IP>:/home/rpi15/edge_nav/models/yolov8n_pruned.hef
```

Then follow [`../../RPI-Rpi5 deployment/README.md`](../../RPI-Rpi5%20deployment/README.md) for the full run guide.

---

## Model Performance

| Metric | Value |
|---|---|
| Architecture | YOLOv8n (pruned, 20% L1 filter removal) |
| Classes | 20 indoor COCO classes |
| mAP50 (Colab T4) | **0.6793** (+9.4% over FP32 baseline) |
| Inference FPS on RPi5 + Hailo-8 | **24.7 FPS** |
| Hailo clusters used | 8 / 8 |

---

*To recompile from scratch, see [`../README.md`](../README.md) — Steps 2 & 3.*
