# Step 1 — Arduino Nicla Vision IMU Setup

This folder contains the MicroPython firmware (`main_Nicla.py`) for the **Arduino Nicla Vision** that streams live gyroscope yaw angles to the Raspberry Pi 5 over USB serial.

---

## What This Does

The Nicla Vision runs on **OpenMV IDE** (MicroPython). On power-up it:

1. Initialises the onboard **LSM6DSOX IMU** (gyroscope + accelerometer)
2. Collects **300 gyro samples** over ~3 seconds to compute and remove the static bias (drift offset)
3. Applies a **dead-band filter (±0.8 °/s)** — ignores micro-vibrations when the device is stationary
4. Streams the integrated yaw angle continuously over USB VCP at **115200 baud** in the format:

```
A:156.68
A:157.12
A:157.90
...
```

The RPi5 reads this stream from `/dev/ttyACM0` in a background thread and uses the yaw angle for sensor fusion in the 2D semantic map.

---

## Hardware Required

| Component | Detail |
|---|---|
| Arduino Nicla Vision | Main board (MicroPython / OpenMV) |
| LSM6DSOX IMU | Onboard — gyroscope + accelerometer |
| micro-USB cable | Nicla Vision → RPi5 USB port |

---

## Software Required

- **OpenMV IDE** — download from [openmv.io/pages/download](https://openmv.io/pages/download)  
  (Available for Windows, macOS, Linux)

---

## Setup Instructions

### 1. Install OpenMV IDE

Download and install OpenMV IDE from [openmv.io](https://openmv.io/pages/download).

### 2. Flash the Firmware

1. Connect the Nicla Vision to your **laptop** via micro-USB
2. Open **OpenMV IDE**
3. Click the **Connect** button (bottom-left plug icon)
4. Copy `main_Nicla.py` (this folder) to the Nicla Vision's internal storage and rename it to `main.py`  
   *(or use File → Save to OpenMV Cam)*
5. Click **Run** (green play button)
6. Open the **Serial Terminal** panel (Tools → Serial Terminal) and verify you see `A:<angle>` lines scrolling

### 3. Mount on RPi5

1. Disconnect from the laptop
2. Plug the Nicla Vision into one of the **RPi5 USB-A ports** via micro-USB
3. The firmware auto-starts on power-up (no laptop needed)

### 4. Gyro Calibration — Critical!

> **Hold the RPi5 assembly completely still and vertical for ~3 seconds after power-on.**

During this time the firmware collects 300 samples to compute the gyro bias offset. Any movement during calibration causes permanent yaw drift for that session.

The calibration indicator appears in the OpenMV serial terminal as:
```
Calibrating gyro... done. Bias = X.XX
```

---

## USB Protocol

The Nicla Vision also supports two-way commands from the RPi5:

| Command (RPi → Nicla) | Response (Nicla → RPi) | Effect |
|---|---|---|
| `PING` | `PONG` | Connectivity check |
| `ZERO` | `ACK:ZERO` | Reset yaw to 0° |
| `START` | `ACK:START` then `A:<angle>` | Begin streaming |
| `STOP` | `ACK:STOP` | Pause streaming |
| `RESET` | `ACK:RESET` | Software reset |

---

## Serial Port on RPi5

The Nicla Vision appears as `/dev/ttyACM0` on the RPi5. If multiple USB-serial devices are connected, it may be `/dev/ttyACM1` — check with:

```bash
ls /dev/ttyACM*
```

Update `SERIAL_PORT` in `main_RPI.py` accordingly.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| No `A:` lines in serial monitor | Firmware not running | Re-flash and click Run in OpenMV IDE |
| Yaw drifts immediately | Moved during calibration | Power-cycle and hold still for 3 s |
| `/dev/ttyACM0` not found on RPi5 | Cable or driver issue | Try a different USB cable; check `dmesg | tail` |
| Yaw jumps erratically | Dead-band too tight | Increase `GYRO_DEAD_BAND` in `main_Nicla.py` |

---

## Files

| File | Description |
|---|---|
| `main_Nicla.py` | MicroPython firmware — flash this to the Nicla Vision as `main.py` |

---

*For the next step (Colab training & HEF compilation), see [`../Colab Development flow/README.md`](../Colab%20Development%20flow/README.md)*
