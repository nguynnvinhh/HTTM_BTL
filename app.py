import os
import glob
import time

from flask import Flask, Response, request, jsonify
from flask_cors import CORS

from src.detector import SafetyDetectorService, VIDEO_EXTS, IMAGE_EXTS

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

detector = SafetyDetectorService()

INPUT_DIR = "input"
TEMP_UPLOAD_DIR = os.path.join("output", "_temp_upload")
os.makedirs(INPUT_DIR, exist_ok=True)

ALLOWED_EXTS = VIDEO_EXTS | IMAGE_EXTS


def list_input_files():
    """Lấy danh sách ảnh và video mẫu có sẵn trong thư mục input."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    return [
        filename
        for filename in sorted(os.listdir(INPUT_DIR))
        if os.path.splitext(filename)[1].lower() in ALLOWED_EXTS
    ]


def make_response(ok, message, status_code=200, **kwargs):
    data = {"ok": ok, "message": message}
    data.update(kwargs)
    return jsonify(data), status_code


@app.route("/video_feed", methods=["GET"])
def video_feed():
    def generate():
        while True:
            try:
                frame_bytes = detector.get_jpeg_frame()
                if frame_bytes is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Cache-Control: no-cache\r\n\r\n"
                        + frame_bytes
                        + b"\r\n"
                    )
                time.sleep(0.03)
            except GeneratorExit:
                break
            except Exception as error:
                print("[VIDEO STREAM ERROR]", error)
                time.sleep(0.1)

    response = Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/api/files", methods=["GET"])
def api_get_files():
    try:
        return jsonify({"ok": True, "files": list_input_files()})
    except Exception as error:
        return make_response(False, f"Lỗi lấy danh sách file: {error}", 500)


@app.route("/api/cameras", methods=["GET"])
def api_get_cameras():
    """Quét và trả về danh sách cổng camera khả dụng trên máy."""
    try:
        return jsonify({"ok": True, "cameras": detector.list_available_cameras()})
    except Exception as error:
        return make_response(False, f"Lỗi quét camera: {error}", 500)


@app.route("/api/source/camera", methods=["POST"])
def api_start_camera():
    """Kết nối Camera Laptop hoặc Iriun Webcam theo camera_index."""
    data = request.get_json(silent=True) or {}
    camera_index = data.get("camera_index", data.get("cam_id", 0))

    try:
        camera_index = int(camera_index)
        if camera_index < 0:
            return make_response(False, "Camera index phải >= 0.", 400)

        ok, message = detector.start_camera(camera_index=camera_index)
        return make_response(ok, message, 200 if ok else 400)
    except ValueError:
        return make_response(False, "Camera index không hợp lệ.", 400)
    except Exception as error:
        print("[CAMERA ERROR]", error)
        return make_response(False, f"Lỗi kết nối camera: {error}", 500)


@app.route("/api/source/iriun", methods=["POST"])
def api_start_iriun():
    """Alias tương thích gọi chung vào hàm mở camera theo index."""
    return api_start_camera()


@app.route("/api/source/existing", methods=["POST"])
def api_start_existing():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "")
    if not filename:
        return make_response(False, "Vui lòng chọn file.", 400)

    safe_filename = os.path.basename(filename)
    filepath = os.path.join(INPUT_DIR, safe_filename)

    if not os.path.isfile(filepath):
        return make_response(False, "Không tìm thấy file.", 404)

    if os.path.splitext(filepath)[1].lower() not in ALLOWED_EXTS:
        return make_response(False, "Định dạng file không được hỗ trợ.", 400)

    try:
        ok, message = detector.start_file(filepath)
        return make_response(ok, message, 200 if ok else 400)
    except Exception as error:
        print("[FILE ERROR]", error)
        return make_response(False, f"Lỗi chạy file: {error}", 500)


@app.route("/api/source/upload", methods=["POST"])
def api_upload_file():
    """Lưu file upload vào thư mục tạm (tự động xóa bản cũ) để không làm rác thư mục input."""
    if "file" not in request.files:
        return make_response(False, "Không có file được gửi lên.", 400)

    file = request.files["file"]
    if not file.filename:
        return make_response(False, "Tên file không hợp lệ.", 400)

    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in ALLOWED_EXTS:
        return make_response(False, "Chỉ hỗ trợ ảnh hoặc video.", 400)

    try:
        detector.stop()

        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        for old_file in glob.glob(os.path.join(TEMP_UPLOAD_DIR, "*")):
            try:
                os.remove(old_file)
            except OSError:
                pass

        save_path = os.path.join(TEMP_UPLOAD_DIR, f"current_upload{extension}")
        file.save(save_path)

        ok, message = detector.start_file(save_path)
        if not ok:
            try:
                os.remove(save_path)
            except OSError:
                pass
            return make_response(False, message, 400)

        prefix = "Ảnh" if extension in IMAGE_EXTS else "Video"
        detector.source_label = f"{prefix}: {file.filename}"
        message = f"Đang xử lý {file.filename}"

        return make_response(
            True,
            message,
            200,
            filename=file.filename,
            files=list_input_files(),
        )
    except Exception as error:
        print("[UPLOAD ERROR]", error)
        return make_response(False, f"Lỗi upload file: {error}", 500)


@app.route("/api/machine_zone", methods=["POST"])
def api_toggle_machine_zone():
    try:
        state = detector.toggle_machine_zone()
        return jsonify({"ok": True, "machine_zone": state})
    except Exception as error:
        print("[MACHINE ZONE ERROR]", error)
        return make_response(False, f"Lỗi thay đổi vùng máy: {error}", 500)


@app.route("/api/stop", methods=["POST"])
def api_stop():
    try:
        detector.stop()
        return jsonify({"ok": True, "message": "Đã dừng giám sát."})
    except Exception as error:
        print("[STOP ERROR]", error)
        return make_response(False, f"Lỗi dừng hệ thống: {error}", 500)


@app.route("/api/status", methods=["GET"])
def api_status():
    try:
        return jsonify(detector.get_status())
    except Exception as error:
        print("[STATUS ERROR]", error)
        return jsonify({
            "mode": "error",
            "source_label": "Lỗi hệ thống",
            "fps": 0,
            "machine_zone": False,
            "warnings": [],
            "logs": [],
            "error": str(error),
        }), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "ok": True,
        "name": "Workplace Safety Detection API",
        "message": "Backend Flask đang hoạt động.",
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({"ok": False, "message": "API không tồn tại."}), 404


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"ok": False, "message": "File tải lên vượt quá giới hạn 500 MB."}), 413


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"ok": False, "message": "Lỗi máy chủ nội bộ."}), 500


if __name__ == "__main__":
    print("=" * 55)
    print(" WORKPLACE SAFETY DETECTION SYSTEM")
    print("=" * 55)
    print("Backend: http://127.0.0.1:5000")
    print("Video:   http://127.0.0.1:5000/video_feed")
    print("=" * 55)

    try:
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        print("\nĐang dừng hệ thống...")
    finally:
        detector.stop()
        print("Đã dừng Backend.")