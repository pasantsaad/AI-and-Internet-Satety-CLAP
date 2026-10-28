"""Serve the laptop's camera as an MJPEG stream on a trusted local network."""

import os
import time

import cv2
from flask import Flask, Response, jsonify

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))

app = Flask(__name__)
camera = None


def get_camera():
    """Open the camera lazily, so Flask can start with a clear error message."""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(CAMERA_INDEX)
        if not camera.isOpened():
            camera.release()
            camera = None
            raise RuntimeError(
                f"Could not open camera index {CAMERA_INDEX}. "
                "Try CAMERA_INDEX=1 or close other apps using the camera."
            )
    return camera


def generate_frames():
    while True:
        try:
            cam = get_camera()
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.1)
                continue

            ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )
        except RuntimeError as exc:
            app.logger.error("%s", exc)
            time.sleep(1)


@app.get("/")
def index():
    return (
        "Laptop camera server is running. Open /video to view the MJPEG stream. "
        "This prototype has no authentication; keep it on a trusted LAN."
    )


@app.get("/health")
def health():
    try:
        cam = get_camera()
        return jsonify(status="ok", camera_open=cam.isOpened(), camera_index=CAMERA_INDEX)
    except RuntimeError as exc:
        return jsonify(status="error", message=str(exc)), 503


@app.get("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


if __name__ == "__main__":
    print(f"Starting camera server on http://{HOST}:{PORT}/video")
    print("Keep this service on a trusted local network; it has no authentication.")
    try:
        app.run(host=HOST, port=PORT, threaded=True, debug=False)
    finally:
        if camera is not None:
            camera.release()
