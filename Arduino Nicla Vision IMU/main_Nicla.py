# main.py for Arduino Nicla Vision / OpenMV
# Two-way USB protocol with Raspberry Pi:
# RPi -> PING   Nicla -> PONG
# RPi -> ZERO   Nicla -> ACK:ZERO
# RPi -> START  Nicla -> ACK:START, then A:<angle>
# RPi -> STOP   Nicla -> ACK:STOP
# RPi -> RESET  Nicla -> ACK:RESET, then software reset

import time
import machine
import pyb
from machine import Pin, SPI
from lsm6dsox import LSM6DSOX

# -----------------------------
# USB VCP
# -----------------------------
usb = pyb.USB_VCP()
cmd_buf = b""

def usb_send(msg):
    try:
        usb.write(msg)
    except Exception:
        pass

def read_command():
    global cmd_buf

    try:
        n = usb.any()
    except Exception:
        return None

    if n <= 0:
        return None

    try:
        cmd_buf += usb.recv(n, timeout=0)
    except Exception:
        return None

    if b"\n" not in cmd_buf:
        return None

    line, cmd_buf = cmd_buf.split(b"\n", 1)

    try:
        return line.decode("utf-8", errors="ignore").strip()
    except Exception:
        return None

# -----------------------------
# IMU init
# -----------------------------
usb_send("BOOT:NICLA_IMU\n")
usb_send("CALIBRATING:keep_still_3s\n")

spi = SPI(5, baudrate=10_000_000, polarity=0, phase=0)
cs = Pin("PF6", Pin.OUT_PP, Pin.PULL_UP)
imu = LSM6DSOX(spi, cs)

# -----------------------------
# Gyro bias calibration
# -----------------------------
n_samples = 300
gz_bias = 0.0

for _ in range(n_samples):
    _, _, gz = imu.gyro()
    gz_bias += gz
    time.sleep_ms(10)

gz_bias /= n_samples

usb_send("BIAS:{:.4f}\n".format(gz_bias))
usb_send("READY\n")

# -----------------------------
# State
# -----------------------------
yaw = 0.0
DEADBAND = 0.8
prev_us = time.ticks_us()
streaming = False

# -----------------------------
# Main loop
# -----------------------------
while True:
    cmd = read_command()

    if cmd:
        if cmd == "PING":
            usb_send("PONG\n")

        elif cmd == "START":
            streaming = True
            prev_us = time.ticks_us()
            usb_send("ACK:START\n")

        elif cmd == "STOP":
            streaming = False
            usb_send("ACK:STOP\n")

        elif cmd == "ZERO":
            yaw = 0.0
            prev_us = time.ticks_us()
            usb_send("ACK:ZERO\n")

        elif cmd == "RESET":
            usb_send("ACK:RESET\n")
            time.sleep_ms(200)
            machine.reset()

        else:
            usb_send("ERR:UNKNOWN_CMD:{}\n".format(cmd))

    if streaming:
        now_us = time.ticks_us()
        dt_s = time.ticks_diff(now_us, prev_us) / 1_000_000.0
        prev_us = now_us

        try:
            _, _, gz = imu.gyro()
        except Exception:
            time.sleep_ms(20)
            continue

        gz_corrected = gz - gz_bias

        if abs(gz_corrected) < DEADBAND:
            gz_corrected = 0.0

        yaw = (yaw + gz_corrected * dt_s) % 360.0

        usb_send("A:{:.2f}\n".format(yaw))

    time.sleep_ms(20)