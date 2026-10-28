"""Run YOLO on a network camera stream without requiring a GUI display."""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Camera stream URL, e.g. http://LAPTOP_IP:5000/video")
    parser.add_argument("--model", default="best.pt", help="Path to trained YOLO weights")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save-dir", default="output", help="Directory for latest annotated frame")
    parser.add_argument("--save-every", type=float, default=2.0, help="Seconds between latest.jpg updates")
    parser.add_argument("--display", action="store_true", help="Show a window; requires a graphical desktop")
    parser.add_argument("--device", default=None, help="Optional inference device")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}. Copy your trained best.pt here or pass --model."
        )

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open stream: {args.source}. Test the URL in a browser on the Pi "
            "and verify that the laptop camera server is running."
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    latest_path = save_dir / "latest.jpg"
    last_save = 0.0
    last_log = 0.0
    frame_count = 0
    start = time.time()

    print(f"Reading stream: {args.source}")
    print(f"Saving annotated frame periodically to: {latest_path}")
    print("Press Ctrl+C to stop. Use --display only in a desktop GUI session.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No frame received; retrying...")
                time.sleep(0.2)
                continue

            kwargs = {"conf": args.conf, "imgsz": args.imgsz, "verbose": False}
            if args.device:
                kwargs["device"] = args.device

            result = model.predict(source=frame, **kwargs)[0]
            annotated = result.plot()
            frame_count += 1
            now = time.time()

            if now - last_save >= args.save_every:
                if not cv2.imwrite(str(latest_path), annotated):
                    print(f"Warning: could not write {latest_path}")
                last_save = now

            if now - last_log >= 2.0:
                detections = []
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        class_name = result.names.get(class_id, str(class_id))
                        detections.append(f"{class_name} {confidence:.2f}")
                elapsed = max(now - start, 0.001)
                fps = frame_count / elapsed
                print(f"FPS(avg): {fps:.1f} | " + (", ".join(detections) if detections else "No target objects"))
                last_log = now

            if args.display:
                cv2.imshow("Smart Trash Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nStopping inference...")
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
