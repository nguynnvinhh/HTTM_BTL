import os
import csv
import cv2
import time
import threading
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from .camera_input import Camera
from .audio_monitor import AudioDistressMonitor


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SafetyDetectorService:
    def __init__(self):
        self.model_fall_path = os.path.join("models", "best.pt")
        self.pose_backbone_path = os.path.join("models", "yolov8s-pose.pt")
        self.vosk_model_path = os.path.join("models", "vosk-vn")

        for path in (self.model_fall_path, self.pose_backbone_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Khong tim thay mo hinh: {path}")

        self.fall_model = YOLO(self.model_fall_path)
        self.pose_model = YOLO(self.pose_backbone_path)

        # Tham so nguong
        self.pose_conf = 0.30
        self.fall_conf = 0.45
        self.wrist_conf = 0.25
        self.alert_after_frames = 2
        self.fall_alert_after_frames = 5
        self.audio_alert_display_seconds = 6

        # Trang thai he thong
        self.mode = "idle"          # "idle" | "camera" | "video" | "image"
        self.source_path = None
        self.source_label = "Chua chon nguon"
        self.enable_machine_zone = True
        self.preferred_camera_id = 1

        # Bien dem & FPS
        self.danger_frames = 0
        self.fall_frames = 0
        self.fps_value = 0.0
        self._fps_prev_time = time.monotonic()

        # Trang thai canh bao am thanh
        self.audio_alert_active = False
        self.audio_alert_reason = ""
        self.audio_alert_time = 0.0
        self.audio_lock = threading.Lock()

        # Frame JPEG xuat ra Web
        self.latest_jpeg = None
        self.frame_lock = threading.Lock()
        self.need_reprocess_image = False

        # Log lich su hien thi len Web
        self.current_warnings = []
        self.recent_logs = []
        self._last_log_times = {}

        os.makedirs("output", exist_ok=True)
        self.log_file = os.path.join(
            "output",
            "alert_log_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv",
        )
        with open(self.log_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "alert_type", "detail"])

        # Quan ly luong chay nen
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.camera = None
        self.video_cap = None
        self.video_writer = None
        self.audio_monitor = None
        self.raw_image = None

        self._render_placeholder("He thong san sang. Hay chon Camera hoac File.")

    def _render_placeholder(self, text: str):
        blank = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.putText(
            blank,
            text,
            (40, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )
        ok, buf = cv2.imencode(".jpg", blank)
        if ok:
            with self.frame_lock:
                self.latest_jpeg = buf.tobytes()

    def log_alert(self, alert_type: str, detail: str = "", cooldown_sec: float = 3.0):
        now = time.monotonic()
        if now - self._last_log_times.get(alert_type, 0) < cooldown_sec:
            return
        self._last_log_times[alert_type] = now

        timestamp = datetime.now().strftime("%H:%M:%S")
        full_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.recent_logs.insert(
            0, {"time": timestamp, "type": alert_type, "detail": detail}
        )
        self.recent_logs = self.recent_logs[:30]

        try:
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([full_ts, alert_type, detail])
        except Exception as exc:
            print("[LOG ERROR]", exc)

    def _on_audio_alert(self, reason, confidence=0.0):
        with self.audio_lock:
            self.audio_alert_active = True
            self.audio_alert_reason = str(reason)
            self.audio_alert_time = time.monotonic()
        self.log_alert("AUDIO_DISTRESS", str(reason), cooldown_sec=4.0)

    def _init_working_camera(self):
        search_ids = [self.preferred_camera_id] + [
            i for i in range(5) if i != self.preferred_camera_id
        ]
        for cam_id in search_ids:
            try:
                cam = Camera(cam_id)
                test_frame = cam.read()
                if test_frame is not None:
                    self.preferred_camera_id = cam_id
                    return cam, cam_id
                cam.release()
            except Exception:
                continue
        return None, None

    def stop(self):
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)

        if self.audio_monitor is not None:
            try:
                self.audio_monitor.stop()
            except Exception:
                pass
            self.audio_monitor = None

        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
            self.camera = None

        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.mode = "idle"
        self.current_warnings = []
        self.fps_value = 0.0
        self._render_placeholder("Da dung giam sat.")

    def start_camera(self):
        self.stop()
        cam, cam_id = self._init_working_camera()
        if cam is None:
            self._render_placeholder("LOI: Khong tim thay Camera hoat dong!")
            return False, "Không tìm thấy camera khả dụng."

        self.camera = cam
        self.mode = "camera"
        self.source_label = f"Camera trực tiếp (Index {cam_id})"
        self.enable_machine_zone = True
        self.danger_frames = 0
        self.fall_frames = 0

        if os.path.isdir(self.vosk_model_path):
            self.audio_monitor = AudioDistressMonitor(
                model_path=self.vosk_model_path,
                alert_callback=self._on_audio_alert,
            )
            self.audio_monitor.start()

        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.worker_thread.start()
        return True, f"Đã kết nối Camera {cam_id}"

    def start_file(self, filepath: str):
        self.stop()
        if not os.path.isfile(filepath):
            return False, f"Không tìm thấy file: {filepath}"

        ext = os.path.splitext(filepath)[1].lower()
        filename = os.path.basename(filepath)
        self.source_path = filepath
        self.enable_machine_zone = False
        self.danger_frames = 0
        self.fall_frames = 0

        if ext in IMAGE_EXTS:
            img_np = np.fromfile(filepath, dtype=np.uint8)
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if frame is None:
                return False, "Không đọc được file ảnh."

            h, w = frame.shape[:2]
            scale = min(1280 / w, 720 / h, 1.0)
            if scale < 1.0:
                frame = cv2.resize(
                    frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
                )

            self.raw_image = frame
            self.mode = "image"
            self.source_label = f"Ảnh: {filename}"
            self.need_reprocess_image = True

        elif ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                return False, "Không mở được file video."

            fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = os.path.join("output", "out_" + filename)

            self.video_cap = cap
            self.video_writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h)
            )
            self.mode = "video"
            self.source_label = f"Video: {filename}"
        else:
            return False, "Định dạng file không được hỗ trợ."

        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.worker_thread.start()
        return True, f"Đang xử lý {filename}"

    def toggle_machine_zone(self):
        self.enable_machine_zone = not self.enable_machine_zone
        if self.mode == "image":
            self.need_reprocess_image = True
        return self.enable_machine_zone

    def _get_machine_region(self, frame):
        h, w = frame.shape[:2]
        return int(0.25 * w), int(0.30 * h), int(0.98 * w), int(0.68 * h)

    def _is_laying_by_geometry(self, keypoints, bbox):
        x1, y1, x2, y2 = bbox
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        aspect_ratio = box_w / float(box_h)

        if keypoints is not None:
            try:
                kpts = keypoints.data[0].cpu().numpy()
                if len(kpts) >= 17:
                    l_hip, r_hip = kpts[11], kpts[12]
                    l_sh, r_sh = kpts[5], kpts[6]
                    hip_conf = (l_hip[2] + r_hip[2]) / 2.0
                    sh_conf = (l_sh[2] + r_sh[2]) / 2.0

                    if sh_conf >= 0.20 and hip_conf >= 0.20:
                        dx = abs(((l_sh[0] + r_sh[0]) / 2.0) - ((l_hip[0] + r_hip[0]) / 2.0))
                        dy = abs(((l_sh[1] + r_sh[1]) / 2.0) - ((l_hip[1] + r_hip[1]) / 2.0))
                        if dx > dy * 1.05:
                            return True
            except Exception:
                pass

        return aspect_ratio > 1.20

    def _process_single_frame(self, frame, is_static_image=False):
        now = time.monotonic()
        self.fps_value = 1.0 / max(now - self._fps_prev_time, 1e-6)
        self._fps_prev_time = now

        pose_result = self.pose_model.predict(
            source=frame, imgsz=416, conf=self.pose_conf, verbose=False
        )[0]
        annotated = pose_result.plot(boxes=False)

        x1_m, y1_m, x2_m, y2_m = self._get_machine_region(frame)
        if self.enable_machine_zone:
            cv2.rectangle(annotated, (x1_m, y1_m), (x2_m, y2_m), (0, 165, 255), 2)
            cv2.putText(
                annotated, "Machine Zone", (x1_m + 4, y1_m + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1
            )

        fall_result = self.fall_model.predict(
            source=frame, imgsz=416, conf=self.fall_conf, verbose=False
        )[0]

        fall_boxes = []
        if fall_result.boxes is not None:
            names = fall_result.names
            for box in fall_result.boxes:
                cid = int(box.cls[0].item())
                cname = str(names.get(cid, cid) if isinstance(names, dict) else names[cid]).lower().strip()
                if cname in {"laying", "fall", "fallen", "falling"}:
                    fall_boxes.append(tuple(int(v) for v in box.xyxy[0].cpu().tolist()))

        raw_fall = False
        person_fall_flags = []

        if pose_result.boxes is not None:
            for idx, box in enumerate(pose_result.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
                kpts = pose_result.keypoints[idx] if pose_result.keypoints is not None else None
                by_geo = self._is_laying_by_geometry(kpts, (x1, y1, x2, y2))
                by_mod = any(
                    not (x2 < fx1 or x1 > fx2 or y2 < fy1 or y1 > fy2)
                    for fx1, fy1, fx2, fy2 in fall_boxes
                )
                falling = by_geo or by_mod
                person_fall_flags.append(falling)
                if falling:
                    raw_fall = True

        if is_static_image:
            fall_detected = raw_fall
        else:
            if raw_fall:
                self.fall_frames += 1
            else:
                self.fall_frames = max(0, self.fall_frames - 1)
            fall_detected = self.fall_frames >= self.fall_alert_after_frames

        if pose_result.boxes is not None:
            for idx, box in enumerate(pose_result.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
                conf = float(box.conf[0].item())
                if fall_detected and person_fall_flags[idx]:
                    color, label = (0, 0, 255), "WARNING: FALL / LYING"
                else:
                    color, label = (0, 255, 0), f"Person ({conf:.2f})"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated, label, (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

        # Kiem tra co tay gan vung may
        hand_risk = False
        if self.enable_machine_zone and pose_result.keypoints is not None:
            for p_idx in range(len(pose_result.keypoints)):
                pts = pose_result.keypoints[p_idx].data[0].cpu().numpy()
                for w_idx in (9, 10):
                    px_f, py_f, w_conf = pts[w_idx][:3]
                    if w_conf >= self.wrist_conf:
                        px, py = int(px_f), int(py_f)
                        if x1_m <= px <= x2_m and y1_m <= py <= y2_m:
                            hand_risk = True
                            cv2.circle(annotated, (px, py), 12, (0, 0, 255), -1)
                            cv2.putText(
                                annotated, "WRIST NEAR MACHINE",
                                (max(5, px - 150), max(25, py - 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                            )

        if self.enable_machine_zone:
            if is_static_image:
                hand_alert = hand_risk
            else:
                self.danger_frames = (self.danger_frames + 1) if hand_risk else 0
                hand_alert = self.danger_frames >= self.alert_after_frames
        else:
            self.danger_frames = 0
            hand_alert = False

        with self.audio_lock:
            sound_active = (
                self.audio_alert_active
                and (time.monotonic() - self.audio_alert_time <= self.audio_alert_display_seconds)
            )
            sound_reason = self.audio_alert_reason

        warnings = []
        if hand_alert:
            warnings.append("WARNING: HAND NEAR MACHINE")
            self.log_alert("HAND_NEAR_MACHINE", "Co tay xam nhap vung may")
        if fall_detected:
            warnings.append("WARNING: FALL / LYING")
            self.log_alert("FALL_DETECTED", "Phat hien nguoi nga / nam san")
        if sound_active:
            warnings.append(f"CANH BAO AM THANH: {sound_reason}")

        self.current_warnings = warnings

        if warnings:
            bh = min(35 * len(warnings) + 12, annotated.shape[0] - 1)
            bw = min(annotated.shape[1] - 1, 900)
            cv2.rectangle(annotated, (0, 0), (bw, bh), (0, 0, 220), -1)
            for i, w_text in enumerate(warnings):
                cv2.putText(
                    annotated, w_text[:90], (10, 28 + i * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
                )

        return annotated

    def _publish_frame(self, annotated):
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            with self.frame_lock:
                self.latest_jpeg = buf.tobytes()

    def _run_loop(self):
        while not self.stop_event.is_set():
            if self.mode == "image":
                if self.need_reprocess_image and self.raw_image is not None:
                    out = self._process_single_frame(self.raw_image.copy(), is_static_image=True)
                    self._publish_frame(out)
                    out_path = os.path.join("output", "out_" + os.path.basename(self.source_path))
                    ext = os.path.splitext(out_path)[1]
                    ok, buf = cv2.imencode(ext, out)
                    if ok:
                        buf.tofile(out_path)
                    self.need_reprocess_image = False
                time.sleep(0.1)

            elif self.mode == "video":
                ret, frame = self.video_cap.read()
                if not ret:
                    if self.video_writer:
                        self.video_writer.release()
                        self.video_writer = None
                    self.mode = "idle"
                    self.source_label += " (Hoàn tất)"
                    break
                out = self._process_single_frame(frame, is_static_image=False)
                if self.video_writer:
                    self.video_writer.write(out)
                self._publish_frame(out)

            elif self.mode == "camera":
                frame = self.camera.read()
                if frame is None:
                    self._render_placeholder("Mat ket noi Camera.")
                    break
                out = self._process_single_frame(frame, is_static_image=False)
                self._publish_frame(out)

    def get_jpeg_frame(self):
        with self.frame_lock:
            return self.latest_jpeg

    def get_status(self):
        return {
            "mode": self.mode,
            "source_label": self.source_label,
            "fps": round(self.fps_value, 1) if self.mode in ("camera", "video") else 0,
            "machine_zone": self.enable_machine_zone,
            "warnings": self.current_warnings,
            "logs": self.recent_logs[:15],
        }