# Smart Trash Detection with YOLO and Raspberry Pi 5

A computer-vision project that detects selected waste items. Training is performed on a laptop; the trained model is then used for inference on a Raspberry Pi 5. The laptop's built-in camera can act as the video source and stream frames over a trusted local network.

**Status:** Starter repository. The code is provided, but the dataset, class mapping, model training, and Raspberry Pi integration must still be tested on your hardware.

## Target classes

This starter project uses these canonical names:

1. `plastic_bottle`
2. `can`
3. `paper`
4. `cardboard`
5. `glass_bottle`

These names must match the labels used by your training dataset. A public dataset may use different class names or IDs. Inspect and remap its labels before training; do not simply change `data.yaml` if the annotation IDs do not match.

## Architecture

```text
Laptop built-in camera
        |
        v
camera/camera_server.py  -- MJPEG over trusted LAN -->
        |
        v
Raspberry Pi 5: inference/detect_stream.py
        |
        v
YOLO detections: class, confidence, bounding box
```

Training runs on the laptop:

```text
Dataset + labels --> training/train.py --> runs/detect/smart-trash/weights/best.pt
```

## Repository layout

```text
smart-trash-detection/
├── README.md
├── .gitignore
├── requirements-laptop.txt
├── requirements-pi.txt
├── camera/
│   └── camera_server.py
├── training/
│   ├── train.py
│   ├── inspect_dataset.py
│   └── data.yaml
├── inference/
│   └── detect_stream.py
├── datasets/
│   ├── public/       # Keep downloaded dataset files out of Git
│   └── custom/       # Keep raw images/labels out of Git unless intentionally small
└── docs/
    └── notes.md
```

## 1. Laptop setup (Fedora/Linux)

From the repository root:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-laptop.txt
```

Check Ultralytics:

```bash
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

If camera access fails, close apps that may already be using the camera and check the Linux camera permissions/device.

## 2. Start the laptop camera stream

From the repository root, activate the virtual environment and run:

```bash
python camera/camera_server.py
```

By default, the server listens on port `5000`. Find the laptop's local IP address and test the stream in a browser:

```text
http://LAPTOP_IP:5000/video
```

Both devices must be able to reach each other on the same trusted LAN. You may need to allow TCP port 5000 through the laptop firewall for the local network only.

**Security:** this basic camera server has no authentication or encryption. Do not expose it to the public internet or configure router port forwarding. Stop it when you are finished.

Optional environment settings:

```bash
CAMERA_INDEX=0 HOST=0.0.0.0 PORT=5000 python camera/camera_server.py
```

## 3. Dataset setup

The file `training/data.yaml` is a **template**. Update its dataset path and confirm the class names/IDs before training. The expected YOLO detection structure is:

```text
datasets/combined/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each image should have a matching `.txt` label file. Each YOLO detection label line is:

```text
class_id x_center y_center width height
```

Coordinates are normalized to the range 0–1. Empty label files are valid for images containing none of the target classes. Dataset YAML paths can be relative to the YAML file or absolute; verify the resolved paths before training.

### Public and custom datasets

The plan is to work on two tracks in parallel:

- **Public dataset:** download an object-detection dataset in YOLO format. One candidate identified during planning is [YOLOv8 Trash Detections on Roboflow Universe](https://universe.roboflow.com/dataset-waste/yolov8-trash-detections-ugh5m). Verify the current version, class list, license, and attribution before use.
- **Custom dataset:** take varied images of plastic bottles, cans, paper, cardboard, and glass bottles. Annotate each target object with a bounding box.

Before combining datasets, map every original class ID to the canonical five-class mapping. Visualize sample annotations and check for duplicates across train/validation/test splits.

Inspect the dataset:

```bash
python training/inspect_dataset.py --data training/data.yaml
```

## 4. Train on the laptop

After editing `training/data.yaml` and confirming labels:

```bash
python training/train.py --data training/data.yaml --model yolo11n.pt --epochs 50 --imgsz 640
```

The pretrained weights may download automatically on first run. Training output is normally created under `runs/detect/smart-trash/`. The best checkpoint is typically:

```text
runs/detect/smart-trash/weights/best.pt
```

The example command is a starting point, not a guarantee of accuracy. Choose epochs, image size, and batch size based on your hardware and dataset. Evaluate on held-out test data and record metrics.

## 5. Copy the model to the Raspberry Pi

From the laptop, for example:

```bash
scp runs/detect/smart-trash/weights/best.pt USER@PI_IP:~/smart-trash-detection/best.pt
```

Replace `USER` and `PI_IP` with your actual Pi username and IP. Do not commit model weights to Git by default.

## 6. Raspberry Pi setup and inference

On Raspberry Pi OS, create a virtual environment:

```bash
mkdir -p ~/smart-trash-detection
cd ~/smart-trash-detection
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-pi.txt
```

Install/copy the repository files to the Pi and place `best.pt` at the repository root. On some Raspberry Pi OS/Python combinations, installing PyTorch/Ultralytics may require version-specific steps. If installation fails, use the exact error to choose compatible versions rather than randomly changing system packages.

Run inference without a GUI (suitable for SSH):

```bash
python inference/detect_stream.py \
  --source http://LAPTOP_IP:5000/video \
  --model best.pt \
  --save-dir output
```

Replace `LAPTOP_IP`. The script saves a periodically updated `output/latest.jpg` and prints detections to the terminal. Download/open the image on the laptop to view it. To open a local display window on a desktop session, add `--display`.

If the stream cannot be opened, first verify `http://LAPTOP_IP:5000/video` from the Pi's browser or use a network diagnostic. `cv2.imshow()` requires a graphical display, so do not use `--display` in a headless SSH session.

## 7. Current checklist

- [ ] Install laptop dependencies
- [ ] Confirm the laptop camera works
- [ ] Start the camera stream and test it from the Pi
- [ ] Download and document the public dataset version/license
- [ ] Capture and annotate custom images
- [ ] Standardize class names and IDs across datasets
- [ ] Inspect dataset labels and visualize examples
- [ ] Train YOLO on the laptop
- [ ] Evaluate on a held-out test set
- [ ] Copy `best.pt` to the Pi
- [ ] Run live inference from the laptop camera stream
- [ ] Record measured FPS, accuracy metrics, and limitations
- [ ] Optional: add LED, buzzer, or servo sorting demo

## Safety, privacy, and licensing

- Do not commit passwords, API keys, private credentials, or sensitive camera footage.
- Large datasets and model weights are normally excluded from Git; link to their source instead.
- Follow each dataset's license and attribution requirements. The repository's code license does not override external dataset licenses.
- Keep the camera stream on a trusted local network.

## Reproducibility log

Record OS/Python versions, package versions, dataset source/version/license, canonical class mapping, training command, model parameters, validation/test metrics, and measured Raspberry Pi inference speed in `docs/notes.md`.
