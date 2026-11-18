import os
import time
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from src.detector import SafetyDetectorService, VIDEO_EXTS, IMAGE_EXTS

app = Flask(__name__)
CORS(app)  # Cho phep React (localhost:5173) goi API sang Flask (localhost:5000)

detector = SafetyDetectorService()
ALLOWED_EXTS = VIDEO_EXTS | IMAGE_EXTS


def list_input_files():
    os.makedirs("input", exist_ok=True)
    return [
        name
        for name in sorted(os.listdir("input"))
        if os.path.splitext(name)[1].lower() in ALLOWED_EXTS
    ]


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame_bytes = detector.get_jpeg_frame()
            if frame_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/files", methods=["GET"])
def api_get_files():
    return jsonify({"ok": True, "files": list_input_files()})


@app.route("/api/source/camera", methods=["POST"])
def api_start_camera():
    ok, msg = detector.start_camera()
    return jsonify({"ok": ok, "message": msg})


@app.route("/api/source/existing", methods=["POST"])
def api_start_existing():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "")
    filepath = os.path.join("input", os.path.basename(filename))
    ok, msg = detector.start_file(filepath)
    return jsonify({"ok": ok, "message": msg})


@app.route("/api/source/upload", methods=["POST"])
def api_upload_file():
    if "file" not in request.files:
        return jsonify({"ok": False, "message": "Không có file được gửi lên."})

    file = request.files["file"]
    if not file.filename:
        return jsonify({"ok": False, "message": "Tên file trống."})

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"ok": False, "message": "Chỉ hỗ trợ file ảnh hoặc video."})

    os.makedirs("input", exist_ok=True)
    safe_name = f"upload_{int(time.time())}{ext}"
    save_path = os.path.join("input", safe_name)
    file.save(save_path)

    ok, msg = detector.start_file(save_path)
    return jsonify({"ok": ok, "message": msg, "files": list_input_files()})


@app.route("/api/machine_zone", methods=["POST"])
def api_toggle_machine_zone():
    state = detector.toggle_machine_zone()
    return jsonify({"ok": True, "machine_zone": state})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    detector.stop()
    return jsonify({"ok": True, "message": "Đã dừng giám sát."})


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify(detector.get_status())


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        detector.stop()