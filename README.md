# Stratus

An autonomous AI-driven system that uses computer vision and cloud analytics to perform high-speed triage on decommissioned hardware that instantly grades components for refurbishment, recovery, or disposal.

## Overview

Stratus is an end-to-end robotic sorting pipeline for IT Asset Disposition (ITAD). It detects objects in a workspace via a USB webcam, sends cropped images to AWS Rekognition for material/condition classification, then physically picks and sorts each item into the appropriate bin using a 6-DOF robotic arm with a motorized gripper.

## Hardware

| Component | Model |
|---|---|
| Robotic Arm | Seeed Studio B601 (6-DOF, Damiao DM-J4310 actuators) |
| Gripper | Custom orange jaw mechanism (Damiao motor at CAN ID 7) |
| Camera | Arducam 1080P Low Light USB (index 1) |
| Controller | USB-to-CAN dongle (`/dev/tty.usbmodem*`) |

## Pipeline

1. **Background capture** — 3-second countdown, user clears workspace
2. **Object detection** — frame differencing + adaptive thresholding finds objects
3. **AI grading** — each object crop is CLAHE-enhanced and sent to Rekognition
4. **User confirmation** — Y to pick, N to skip, Q to quit
5. **Pick & place** — arm approaches, descends, grips, lifts, moves to bin, releases
6. **Telemetry** — classification result written to DynamoDB + published via MQTT

## Telemetry

**Cloud**
- **DynamoDB** — persistent table `stratus_classifications` stores every grading result with labels, confidence scores, timestamps, and bin assignments
- **AWS IoT Core** — real-time MQTT topic `stratus/stratus-dev-01/telemetry` streams classifications for live dashboards or downstream consumers
- **AWS Rekognition** — per-crop image analysis generates label metadata and bounding boxes that are logged alongside each record
- **Scalable schema** — cloud storage supports querying by grade, date range, or device for historical trend analysis

**Local**
- **JSONL log** — `data/logs/telemetry.jsonl` maintains an offline append-only backup of every classification, decoupled from network availability
- **Live preview** — OpenCV window shows bounding boxes, current grade, and bin assignment in real time
- **Zero-trust fallback** — if cloud connectivity drops, the pipeline continues operating and the local log is available for later sync or replay

## Tech Stack

- **Language**: Python 3.10
- **Robotics**: Seeed Studio B601 arm, Damiao DM-J4310 motors, USB-to-CAN (motorbridge)
- **Vision**: OpenCV, CLAHE preprocessing, background subtraction, contour detection
- **AI / ML**: AWS Rekognition (label detection, per-crop grading), YOLO-World (on-device)
- **Cloud**: AWS DynamoDB, AWS IoT Core MQTT, AWS IAM
- **Telemetry**: DynamoDB (persistent), IoT Core MQTT (streaming), JSONL (local)
- **Hardware Interface**: USB webcam (Arducam 1080P), serial CAN dongle
- **OS**: macOS (Apple Silicon)

## Efficiency & Performance

The pipeline is tuned to keep detection fast and the shared CAN bus clean:

- **Accelerated YOLO inference** — auto-selects the fastest backend (MPS on Apple
  Silicon, CUDA, else CPU) via `--yolo-device`, with configurable inference
  resolution via `--yolo-imgsz` (lower = faster, default 640).
- **Model warm-up** — one throwaway inference at load caches the graph and
  compiles MPS/CUDA kernels so the live loop never pays the multi-second
  cold-start penalty on the first real frame.
- **Latency telemetry** — every detection logs a rolling inference latency
  (`[infer N ms]`) so speed regressions are visible in the console.
- **Classify throttling** — the model only runs every N frames
  (`--classify-every`, default 3); preview drawing and box overlays run every
  frame without re-running inference.
- **Dedicated gripper MIT stream** — a separate 50 Hz thread drives the gripper
  so its continuous position frames never compete with the 10 Hz arm control
  loop on the shared serial/CAN bus (single-shot MIT frames only pulse a Damiao
  motor — continuous streaming is what actually actuates it).
- **Largest-area pick target** — the primary pick point comes from the
  largest-area bounding box (not the highest-confidence fragment), so tiny
  spurious boxes (handles, rims, shadows) are ignored and the arm reaches the
  true object center.
- **Zero-copy background refresh** — the background reference is recaptured
  after each pick, and non-pick frames skip classifier work entirely.
