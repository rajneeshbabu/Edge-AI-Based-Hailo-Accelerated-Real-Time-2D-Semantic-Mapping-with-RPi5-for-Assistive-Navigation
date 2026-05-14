# Steps 2 & 3 — Colab Training, ONNX Export & HEF Compilation

This folder contains the Jupyter notebook (`EDGE_SLAM.ipynb`) that covers the full model pipeline — from fine-tuning YOLOv8n through INT8 quantization, L1 pruning, and final compilation to a Hailo HEF binary.

The pipeline is split into **two phases** that must run in **separate Colab sessions** (the Hailo DFC is incompatible with Colab's default CUDA/Python environment).

---

## Model Files (Pre-compiled)

The `models/` subfolder in this directory contains all compiled and intermediate model artifacts — **no recompilation needed** to deploy.

| File | Size | Description |
|---|---|---|
| `models/yolov8n_pruned.hef` | **4.3 MB** | ✅ **Deploy this** — Hailo Executable Format, ready for RPi5 + Hailo-8 |
| `models/best_opset11.onnx` | 13 MB | Pruned YOLOv8n ONNX export (opset-11), input to Hailo DFC |
| `models/yolov8n_pruned.har` | 13 MB | Hailo Archive — intermediate quantized format produced by DFC optimizer |
| `models/best_pruned.pt` | 6.3 MB | Pruned YOLOv8n PyTorch checkpoint — use this to resume fine-tuning |

> **To deploy immediately:** copy `models/yolov8n_pruned.hef` to the RPi5 at `~/edge_nav/models/yolov8n_pruned.hef` and follow [`../RPI-Rpi5 deployment/README.md`](../RPI-Rpi5%20deployment/README.md).

---

## What This Produces (if recompiling from scratch)

| Output | Size | Where it goes |
|---|---|---|
| `best_opset11.onnx` | ~13 MB | Google Drive → `edge_ai_project/` → copy to `models/` |
| `yolov8n_pruned.hef` | **4.3 MB** | Download to laptop → copy to `models/` → transfer to RPi5 |

---

## Hardware / Runtime Required

- **Google Colab** with a **T4 GPU** runtime (free tier works)
- A **Google account** with Google Drive
- A **Hailo Developer Zone account** (free) — needed to download the DFC wheel in Phase B

---

## Step 2 — Phase A: Training, Optimization & ONNX Export

### What Phase A covers

The notebook runs 5 optimization stages sequentially:

| Stage | What happens | mAP50 | FPS (T4) |
|---|---|---|---|
| 1 — FP32 Baseline | Fine-tune pretrained YOLOv8n on COCO128 (20 classes, 20 epochs) | 0.6269 | 120.5 |
| 2 — PTQ INT8 | Post-training quantization with 64 calibration images | 0.6093 | 214.4 |
| 3 — QAT INT8 | Quantization-aware training (20 epochs, simulated noise) | 0.5944 | 186.6 |
| 4 — L1 Pruning ✅ | Remove 20% weakest Conv2d filters + 20-epoch fine-tune | **0.6793** | 218.3 |
| 5 — KD Ablation | Knowledge distillation from YOLOv8s teacher (not deployed) | 0.5904 | 120.5 |

> Stage 4 (L1 Pruned) is the **deployed model**. Pruning removes noise-carrying filters, which paradoxically improves mAP over the FP32 baseline by +8.4%.

### Instructions

1. Open `EDGE_SLAM.ipynb` in **Google Colab**
   - Runtime → Change runtime type → **T4 GPU**
2. Mount your Google Drive when prompted  
   (all outputs save to `My Drive/edge_ai_project/`)
3. **Run all cells EXCEPT the last cell**
   - The notebook auto-downloads COCO128 (~6 MB) from Ultralytics
   - Each training stage prints results to the cell output
   - The pruned ONNX model is saved as `best_opset11.onnx` in your Drive
4. **Do NOT run the last cell yet** — it requires a fresh session (see Phase B below)

> **Runtime note:** Phase A takes ~25–40 minutes total on a T4 GPU. Colab may disconnect after 90 minutes of idle — run stages without long breaks.

### Key outputs saved to Google Drive

```
My Drive/edge_ai_project/
├── best_opset11.onnx          ← pruned model in ONNX opset-11 format
├── runs/detect/train*/        ← YOLOv8 training logs & checkpoints
└── (calibration images kept in /content/ during session)
```

---

## Step 3 — Phase B: HEF Compilation

> **A fresh Colab session is mandatory.** The Hailo Dataflow Compiler (DFC) requires Python 3.10 and is incompatible with Colab's default CUDA 12 / Python 3.11 environment. Running Phase B in the same session as Phase A will fail.

### What Phase B does

Compiles `best_opset11.onnx` → Hailo Executable Format (HEF) via:

```
ONNX (opset-11)
    ↓  hailo parser
HAR (Hailo Archive)
    ↓  hailo optimize  (INT8 quantization, 64 calibration images, CPU-only)
HAR (quantized)
    ↓  hailo compiler
HEF (4.5 MB)  ←── deployed on RPi5
```

### Instructions

#### 3a. Download the Hailo DFC wheel

1. Go to [hailo.ai](https://hailo.ai) and create a free developer account (or log in)
2. Navigate: **Developer Zone → Downloads → AI Accelerator → AI Suite → Dataflow Compiler → Linux → x86_64 → Python 3.10**
3. Download `hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl` (~488 MB)
4. Upload the `.whl` to your Google Drive at `My Drive/edge_ai_project/`

#### 3b. Run the last cell

1. In the **same Colab notebook**, go to **Runtime → Restart session** (clears the previous Python env)
2. After the restart, scroll to the **very last cell** and run **only that cell**
3. Grant Google Drive mount permission when prompted
4. The cell will:
   - Create a Python 3.10 virtual environment (takes ~2 minutes)
   - Install the DFC wheel from your Drive
   - Run `hailo parser`, `hailo optimize` (CPU, ~5–8 minutes), and `hailo compile`
   - Save `yolov8n_pruned.hef` to your Drive

#### 3c. Download the HEF

1. Once the cell completes, find `yolov8n_pruned.hef` in your Google Drive (`edge_ai_project/`)
2. Download it to your **laptop** (it's only 4.5 MB)

---

## Why opset-11?

YOLOv8 normally exports to ONNX opset-13, but the Hailo DFC v3.33.1 does not support several opset-13 operators (notably `GridSample` and updated `Resize`). The notebook forces `opset=11` during export:

```python
model.export(format='onnx', opset=11, simplify=True)
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: hailo_sdk_client` | Make sure you're running only the last cell after a **fresh session restart** |
| DFC wheel install fails | Verify the `.whl` filename matches exactly; re-download if corrupted |
| `hailo optimize` runs for >30 minutes | Normal on CPU — let it finish; do not interrupt |
| Drive permission denied | Re-run the Drive mount cell and approve in the popup |
| ONNX export error `opset not supported` | Ensure `opset=11` is set in the export call |

---

## Files in This Folder

| File | Description |
|---|---|
| `EDGE_SLAM.ipynb` | Full pipeline notebook — Phase A (all cells except last) + Phase B (last cell only) |
| `models/yolov8n_pruned.hef` | ✅ Compiled Hailo model — deploy on RPi5 |
| `models/best_opset11.onnx` | Pruned ONNX export (opset-11) used as DFC input |
| `models/yolov8n_pruned.har` | Hailo Archive — INT8-quantized intermediate artifact |
| `models/best_pruned.pt` | Pruned PyTorch checkpoint — resume training from here |

---

*After downloading `yolov8n_pruned.hef`, continue to [`../RPI-Rpi5 deployment/README.md`](../RPI-Rpi5%20deployment/README.md) for RPi5 setup and deployment.*
