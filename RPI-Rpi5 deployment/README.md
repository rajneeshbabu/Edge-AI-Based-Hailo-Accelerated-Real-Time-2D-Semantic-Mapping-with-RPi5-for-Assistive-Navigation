# Steps 4, 5 & 6 — RPi5 Setup, File Transfer & Running the System

This folder contains the main edge-inference script (`main_RPI.py`) that runs entirely on the **Raspberry Pi 5 + Hailo-8 AI HAT**. It handles camera capture, NPU inference, DFL decoding, IMU yaw fusion, and real-time 2D semantic map rendering.

---

## What This Does

`main_RPI.py` runs two OpenCV windows simultaneously on the RPi5:

| Window | Description |
|---|---|
| **Detection Feed** | Live camera feed with YOLOv8 bounding boxes, class labels, confidence scores, and FPS overlay |
| **2D Semantic Map** | Top-down polar map with range rings at 1 m intervals; detected objects placed by heading + distance |

**On-device performance:**

| Metric | Value |
|---|---|
| Inference FPS (RPi5 + Hailo-8) | **24.7 FPS** |
| HEF model size | **4.5 MB** |
| Hailo clusters used | 8 / 8 |
| IMU yaw update rate | 50 Hz |

---

## Prerequisites

- Raspberry Pi 5 (4 GB or 8 GB) running **Raspberry Pi OS Bookworm (64-bit)**
- **Hailo-8 AI HAT** mounted and connected via M.2/PCIe
- **TNBA1392 camera** connected to CAM0 (CSI ribbon)
- **Arduino Nicla Vision** flashed and connected via micro-USB (see `../Arduino Nicla Vision IMU/README.md`)
- `yolov8n_pruned.hef` downloaded from the Colab pipeline (see `../Colab Development flow/README.md`)
- Network access on first setup (for `apt install`)

---

## Step 4 — RPi5 One-Time Environment Setup

Connect to the RPi5 via SSH or open a terminal directly:

```bash
# 1. System packages (camera, serial, OpenCV, Hailo runtime)
sudo apt update
sudo apt install -y python3-venv python3-numpy python3-opencv \
  python3-serial python3-picamera2 v4l-utils rpicam-apps

# 2. Install Hailo runtime (adds hailort and hailo-all)
sudo apt install -y hailo-all

# 3. Create project directory structure
mkdir -p ~/edge_nav/models ~/edge_nav/logs

# 4. Create virtual environment (with system packages so picamera2 / opencv are visible)
cd ~/edge_nav
python3 -m venv venv_edge_nav --system-site-packages

# 5. Activate and install remaining Python packages
source ~/edge_nav/venv_edge_nav/bin/activate
pip install pyserial pyttsx3

# 6. Verify Hailo is detected
python -c "from hailo_platform import HEF, VDevice; print('Hailo OK')"
hailortcli fw-control identify
```

Expected output from `hailortcli fw-control identify`:
```
Identifying board
Control Protocol Version: 2
Firmware Version: 4.17.0 (release,app,extended context switch buffer)
Logger Version: 0
Board Name: Hailo-8
Device Architecture: HAILO8
Serial Number: HLLWM2A...
...
```

If `Hailo OK` prints without error and `hailortcli` returns board info, the hardware is ready.

---

## Step 5 — Transfer Files to RPi5

You need to copy two files from your **laptop** to the RPi5:

| File | Destination on RPi5 |
|---|---|
| `yolov8n_pruned.hef` | `~/edge_nav/models/yolov8n_pruned.hef` |
| `main_RPI.py` (this folder) | `~/edge_nav/main.py` |

### From Windows PowerShell

Replace `<RPI_IP>` with your RPi5's Wi-Fi IP address (find it with `hostname -I` on the RPi5):

```powershell
# Transfer the HEF model
scp "$env:USERPROFILE\Downloads\yolov8n_pruned.hef" `
    rpi15@<RPI_IP>:/home/rpi15/edge_nav/models/yolov8n_pruned.hef

# Transfer the inference script
scp "$env:USERPROFILE\Downloads\main_RPI.py" `
    rpi15@<RPI_IP>:/home/rpi15/edge_nav/main.py
```

### From macOS / Linux Terminal

```bash
scp ~/Downloads/yolov8n_pruned.hef rpi15@<RPI_IP>:/home/rpi15/edge_nav/models/yolov8n_pruned.hef
scp path/to/main_RPI.py rpi15@<RPI_IP>:/home/rpi15/edge_nav/main.py
```

### Verify the transfer

On the RPi5:
```bash
ls -lh ~/edge_nav/models/
# Should show: yolov8n_pruned.hef  ~4.5M
```

---

## Step 6 — Run the System

```bash
# Activate the virtual environment
cd ~/edge_nav
source ~/edge_nav/venv_edge_nav/bin/activate

# Set display (required if running over SSH with X-forwarding, or use :0 for local display)
export DISPLAY=:0

# Run the inference + mapping system
python main.py
```

Two OpenCV windows will open on the RPi5 display:
- **Window 1** — Live camera feed with bounding boxes
- **Window 2** — 2D polar semantic map

Press **`q`** in either window to quit cleanly.

### Running headless (no display / SSH)

Edit `main_RPI.py` and set:
```python
SHOW_WINDOW = False    # disables OpenCV display
ENABLE_AUDIO = True    # pyttsx3 audio announcements still work
```

Then run as normal. Object announcements will play through the RPi5 audio output.

---

## Key Configuration Options in `main_RPI.py`

| Variable | Default | Description |
|---|---|---|
| `HEF_PATH` | `~/edge_nav/models/yolov8n_pruned.hef` | Path to the compiled HEF model |
| `SERIAL_PORT` | `/dev/ttyACM0` | USB serial port for Nicla Vision IMU |
| `SERIAL_BAUD` | `115200` | Serial baud rate |
| `CONF_THRESH` | `0.40` | Minimum detection confidence |
| `IOU_THRESH` | `0.45` | NMS IoU threshold |
| `SHOW_WINDOW` | `True` | Enable/disable OpenCV display |
| `ENABLE_AUDIO` | `True` | Enable/disable pyttsx3 voice announcements |
| `HALF_FOV_DEG` | `30.0` | Half-FOV for camera angle estimation (degrees) |
| `MAX_OBJ_AGE` | `10.0` | Seconds before an unseen object is pruned from the map |

---

## How the Inference Pipeline Works

```
Picamera2 (640×480 RGB)
    ↓
Hailo-8 NPU  ←  yolov8n_pruned.hef
    ↓  6 raw tensors (3 box-dist + 3 class-scores)
DFL Decoder (RPi5 CPU)
    ↓  decoded [x1,y1,x2,y2,conf,cls] per detection
NMS (OpenCV, conf > 0.40, IoU < 0.45)
    ↓  filtered detections
Sensor Fusion
    ↓  angle = yaw_IMU + camera_offset_angle
    ↓  distance = h_ref_class / bbox_height
EMA Smoothing (λ=0.7)
    ↓
2D Polar Map (520×520 px OpenCV canvas)
    + Detection Feed (labeled bounding boxes)
```

### Distance estimation per class

Each class has a reference height (the expected bounding-box pixel height when the object is 1 m from the camera). The distance formula is:

```
d = max(0.3 m,  h_ref[class] / bbox_height_pixels)
```

Objects closer than 0.3 m are clamped to 0.3 m to avoid division artefacts.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `HEF file not found` | Wrong path | Check `HEF_PATH` in `main_RPI.py` |
| `ImportError: hailo_platform` | venv not activated | Run `source ~/edge_nav/venv_edge_nav/bin/activate` |
| Camera fails to open | CSI ribbon not seated | Re-seat ribbon; run `rpicam-hello` to test |
| `serial.SerialException /dev/ttyACM0` | Nicla not connected or wrong port | Check `ls /dev/ttyACM*`; update `SERIAL_PORT` |
| Display not found | `DISPLAY` not set | Run `export DISPLAY=:0` before starting |
| FPS drops below 15 | CPU overloaded | Disable audio (`ENABLE_AUDIO=False`) or reduce map print interval |
| Hailo not detected | HAT not powered | Ensure Hailo AI HAT is fully seated in M.2 slot; reboot RPi5 |

---

## Physical Connections

```
TNBA1392 Camera  ── CSI ribbon ──► RPi5 CAM0
Hailo-8 AI HAT   ── M.2 / PCIe ──► RPi5 (stacked on top)
Nicla Vision     ── micro-USB  ──► RPi5 USB port
Power Bank       ── USB-C      ──► RPi5
```

---

## Files in This Folder

| File | Description |
|---|---|
| `main_RPI.py` | Edge inference script — copy to `~/edge_nav/main.py` on the RPi5 |

---

*For hardware setup and wiring details, see the [main project README](../README.md).*  
*For Nicla Vision IMU firmware, see [`../Arduino Nicla Vision IMU/README.md`](../Arduino%20Nicla%20Vision%20IMU/README.md).*
