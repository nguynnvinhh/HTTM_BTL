import os
import csv
import cv2
import time
import threading
import numpy as np

from datetime import datetime
from ultralytics import YOLO

try:
    from .camera_input import Camera
    from .audio_monitor import AudioDistressMonitor
except ImportError:
    from camera_input import Camera
    from audio_monitor import AudioDistressMonitor


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SafetyDetectorService:

    def __init__(self):
        # 1. ĐƯỜNG DẪN MODEL
        self.model_fall_path = os.path.join("models", "best.pt")
        self.pose_backbone_path = os.path.join("models", "yolov8s-pose.pt")
        self.model_machine_path = os.path.join("models", "machine.pt")
        self.vosk_model_path = os.path.join("models", "vosk-vn")

        for path in (self.model_fall_path, self.pose_backbone_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Không tìm thấy model: {path}")

        print("[INFO] Đang tải model nhận diện...")
        self.fall_model = YOLO(self.model_fall_path)
        self.pose_model = YOLO(self.pose_backbone_path)

        if os.path.isfile(self.model_machine_path):
            self.machine_model = YOLO(self.model_machine_path)
            print("[INFO] Đã tải model nhận diện máy móc.")
        else:
            self.machine_model = None
            print("[WARN] Không có models/machine.pt. Sử dụng vùng máy mặc định.")

        # 2. THAM SỐ NHẬN DIỆN
        self.pose_conf = 0.30
        self.fall_conf = 0.45
        self.machine_conf = 0.35

        self.machine_margin = 25
        self.limb_conf = 0.25

        self.alert_after_frames = 2
        self.fall_alert_after_frames = 5
        self.audio_alert_display_seconds = 6

        # 3. CẤU HÌNH CAMERA & TRẠNG THÁI
        self.mode = "idle"
        self.source_path = None
        self.source_label = "Chưa chọn nguồn"
        self.enable_machine_zone = True
        self.active_camera_id = None

        # 4. CẤU HÌNH MICROPHONE KÉP (Index 4: Iriun, Index 3: Laptop)
        self.audio_devices = [4, 3]
        self.audio_monitor = None
        self.audio_alert_active = False
        self.audio_alert_reason = ""
        self.audio_alert_time = 0.0
        self.audio_lock = threading.Lock()

        # 5. BIẾN ĐẾM VÀ FPS
        self.hand_danger_frames = 0
        self.foot_danger_frames = 0
        self.fall_frames = 0
        self.fps_value = 0.0
        self._fps_prev_time = time.monotonic()

        # 6. FRAME HIỂN THỊ
        self.latest_jpeg = None
        self.frame_lock = threading.Lock()
        self.need_reprocess_image = False
        self.raw_image = None

        # 7. CẢNH BÁO VÀ LOG
        self.current_warnings = []
        self.recent_logs = []
        self._last_log_times = {}
        self.log_lock = threading.Lock()

        os.makedirs("output", exist_ok=True)
        self.log_file = os.path.join(
            "output",
            "alert_log_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv",
        )
        with open(self.log_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "alert_type", "detail"])

        # 8. QUẢN LÝ LUỒNG AN TOÀN
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.camera = None
        self.video_cap = None
        self.video_writer = None
        self._lifecycle_lock = threading.RLock()

        self._render_placeholder("He thong san sang. Hay chon Camera hoac File.")

    def _render_placeholder(self, text):
        blank = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.putText(
            blank, text, (40, 270),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2, cv2.LINE_AA
        )
        ok, buf = cv2.imencode(".jpg", blank)
        if ok:
            with self.frame_lock:
                self.latest_jpeg = buf.tobytes()

    def log_alert(self, alert_type, detail="", cooldown_sec=3.0):
        now = time.monotonic()
        with self.log_lock:
            if now - self._last_log_times.get(alert_type, 0) < cooldown_sec:
                return
            self._last_log_times[alert_type] = now

            timestamp = datetime.now().strftime("%H:%M:%S")
            full_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            self.recent_logs.insert(
                0, {"time": timestamp, "type": alert_type, "detail": detail}
            )
            self.recent_logs = self.recent_logs[:30]

            try:
                with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([full_timestamp, alert_type, detail])
            except Exception as exc:
                print("[LOG ERROR]", exc)

    def _on_audio_alert(self, reason, confidence=0.0):
        with self.audio_lock:
            self.audio_alert_active = True
            self.audio_alert_reason = str(reason)
            self.audio_alert_time = time.monotonic()
        self.log_alert("AUDIO_DISTRESS", str(reason), cooldown_sec=4.0)

    def list_available_cameras(self):
        """Quét các cổng camera khả dụng trên máy (0..3) và gắn nhãn rõ ràng."""
        cameras = []
        for idx in range(4):
            label = (
                "Camera Laptop (Index 0)"
                if idx == 0
                else f"Iriun / Cam ngoài (Index {idx})"
            )
            if self.mode == "camera" and self.active_camera_id == idx and self.camera is not None:
                cameras.append({"id": idx, "label": label})
                continue

            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    cameras.append({"id": idx, "label": label})
            else:
                cap.release()
        return cameras

    def stop(self):
        with self._lifecycle_lock:
            self.stop_event.set()

            if (
                self.worker_thread is not None
                and self.worker_thread.is_alive()
                and threading.current_thread() is not self.worker_thread
            ):
                self.worker_thread.join(timeout=5.0)
            self.worker_thread = None

            if self.audio_monitor is not None:
                try:
                    self.audio_monitor.stop()
                except Exception as exc:
                    print("[AUDIO] Lỗi khi dừng microphone:", exc)
                self.audio_monitor = None

            if self.camera is not None:
                try:
                    self.camera.release()
                except Exception:
                    pass
                self.camera = None

            if self.video_cap is not None:
                try:
                    self.video_cap.release()
                except Exception:
                    pass
                self.video_cap = None

            if self.video_writer is not None:
                try:
                    self.video_writer.release()
                except Exception:
                    pass
                self.video_writer = None

            self.mode = "idle"
            self.active_camera_id = None
            self.current_warnings = []
            self.fps_value = 0.0
            self._render_placeholder("Da dung giam sat.")

    def _start_audio_monitor(self):
        if not os.path.isdir(self.vosk_model_path):
            print("[AUDIO] Không tìm thấy Vosk model:", self.vosk_model_path)
            return False

        try:
            self.audio_monitor = AudioDistressMonitor(
                model_path=self.vosk_model_path,
                alert_callback=self._on_audio_alert,
                device=self.audio_devices,
                sample_rate=16000,
                cooldown=6,
                gain=2.5,
            )
            self.audio_monitor.start()
            return True
        except Exception as exc:
            print("[AUDIO] Không khởi động được microphone:", exc)
            self.audio_monitor = None
            return False

    def start_camera(self, camera_index=0):
        """Mở đúng camera theo index được chọn (0: Laptop, 1+: Iriun / Cam ngoài)."""
        self.stop()

        try:
            camera_index = int(camera_index)
        except (TypeError, ValueError):
            camera_index = 0

        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(camera_index)

        if not capture.isOpened():
            capture.release()
            err_msg = f"Không mở được Camera Index {camera_index}. Hãy kiểm tra kết nối."
            self._render_placeholder(f"LOI: {err_msg}")
            return False, err_msg

        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            err_msg = f"Không đọc được hình ảnh từ Camera Index {camera_index}."
            self._render_placeholder(f"LOI: {err_msg}")
            return False, err_msg

        cam_name = (
            "Camera Laptop"
            if camera_index == 0
            else "Iriun / Cam ngoài"
        )
        label = f"{cam_name} (Index {camera_index})"

        self.camera = capture
        self.active_camera_id = camera_index
        self.mode = "camera"
        self.source_label = label
        self.enable_machine_zone = True

        self.hand_danger_frames = 0
        self.foot_danger_frames = 0
        self.fall_frames = 0
        self.fps_value = 0.0
        self._fps_prev_time = time.monotonic()

        with self.audio_lock:
            self.audio_alert_active = False
            self.audio_alert_reason = ""
            self.audio_alert_time = 0.0

        self._start_audio_monitor()

        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.worker_thread.start()

        return True, f"Đã kết nối {label}"

    def start_iriun_camera(self, camera_index=1):
        return self.start_camera(camera_index=camera_index)

    def start_file(self, filepath):
        self.stop()

        if not os.path.isfile(filepath):
            return False, f"Không tìm thấy file: {filepath}"

        ext = os.path.splitext(filepath)[1].lower()
        filename = os.path.basename(filepath)

        self.source_path = filepath
        self.enable_machine_zone = True
        self.hand_danger_frames = 0
        self.foot_danger_frames = 0
        self.fall_frames = 0
        self.fps_value = 0.0

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
                cap.release()
                return False, "Không mở được file video."

            fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.video_cap = cap
            out_path = os.path.join(
                "output", "out_" + os.path.splitext(filename)[0] + ".mp4"
            )
            self.video_writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h)
            )
            if not self.video_writer.isOpened():
                self.video_writer.release()
                self.video_writer = None
                cap.release()
                self.video_cap = None
                return False, "Không tạo được video đầu ra."

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

    def _get_fallback_machine_region(self, frame):
        h, w = frame.shape[:2]
        return [(int(0.25 * w), int(0.30 * h), int(0.98 * w), int(0.68 * h), 1.0)]

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

        frame_h, frame_w = frame.shape[:2]
        frame_area = max(1, frame_h * frame_w)

        # 1. NHẬN DIỆN NGƯỜI VÀ KHUNG XƯƠNG
        pose_result = self.pose_model.predict(
            source=frame, imgsz=416, conf=self.pose_conf, verbose=False
        )[0]
        annotated = pose_result.plot(boxes=False)

        # 2. NHẬN DIỆN MÁY MÓC
        machine_boxes = []
        if self.enable_machine_zone:
            if self.machine_model is not None:
                machine_result = self.machine_model.predict(
                    source=frame, imgsz=416, conf=self.machine_conf, verbose=False
                )[0]
                if machine_result.boxes is not None:
                    for box in machine_result.boxes:
                        mx1, my1, mx2, my2 = map(int, box.xyxy[0].cpu().tolist())
                        m_conf = float(box.conf[0].item())
                        m_area = max(1, (mx2 - mx1) * (my2 - my1))
                        ratio = m_area / frame_area
                        if 0.03 <= ratio <= 0.85:
                            machine_boxes.append((mx1, my1, mx2, my2, m_conf))
            else:
                machine_boxes = self._get_fallback_machine_region(frame)

            for mx1, my1, mx2, my2, m_conf in machine_boxes:
                cv2.rectangle(annotated, (mx1, my1), (mx2, my2), (0, 165, 255), 2)
                label_m = (
                    f"Machine ({m_conf:.2f})"
                    if self.machine_model is not None
                    else "Machine Zone"
                )
                cv2.putText(
                    annotated, label_m, (mx1 + 4, max(20, my1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2
                )

        # 3. NHẬN DIỆN TÉ NGÃ
        fall_result = self.fall_model.predict(
            source=frame, imgsz=416, conf=self.fall_conf, verbose=False
        )[0]

        fall_boxes = []
        if fall_result.boxes is not None:
            names = fall_result.names
            for box in fall_result.boxes:
                cid = int(box.cls[0].item())
                cname = names.get(cid, str(cid)) if isinstance(names, dict) else names[cid]
                if str(cname).lower().strip() in {"laying", "fall", "fallen", "falling"}:
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

        # 4. KIỂM TRA CỔ TAY (9, 10) VÀ CỔ CHÂN (15, 16) TRONG VÙNG MÁY
        hand_risk = False
        foot_risk = False

        if self.enable_machine_zone and machine_boxes and pose_result.keypoints is not None:
            margin = self.machine_margin if self.machine_model is not None else 0
            for p_idx in range(len(pose_result.keypoints)):
                pts = pose_result.keypoints[p_idx].data[0].cpu().numpy()

                for w_idx in (9, 10):
                    px_f, py_f, conf = pts[w_idx][:3]
                    if conf >= self.limb_conf:
                        px, py = int(px_f), int(py_f)
                        in_zone = any(
                            (mx1 - margin) <= px <= (mx2 + margin)
                            and (my1 - margin) <= py <= (my2 + margin)
                            for mx1, my1, mx2, my2, _ in machine_boxes
                        )
                        if in_zone:
                            hand_risk = True
                            cv2.circle(annotated, (px, py), 10, (0, 0, 255), -1)
                            cv2.putText(
                                annotated, "WRIST NEAR MACHINE",
                                (max(5, px - 110), max(25, py - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
                            )

                for a_idx in (15, 16):
                    px_f, py_f, conf = pts[a_idx][:3]
                    if conf >= self.limb_conf:
                        px, py = int(px_f), int(py_f)
                        in_zone = any(
                            (mx1 - margin) <= px <= (mx2 + margin)
                            and (my1 - margin) <= py <= (my2 + margin)
                            for mx1, my1, mx2, my2, _ in machine_boxes
                        )
                        if in_zone:
                            foot_risk = True
                            cv2.circle(annotated, (px, py), 10, (0, 100, 255), -1)
                            cv2.putText(
                                annotated, "FOOT NEAR MACHINE",
                                (max(5, px - 100), max(25, py - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2
                            )

        if self.enable_machine_zone:
            if is_static_image:
                hand_alert = hand_risk
                foot_alert = foot_risk
            else:
                self.hand_danger_frames = (self.hand_danger_frames + 1) if hand_risk else 0
                self.foot_danger_frames = (self.foot_danger_frames + 1) if foot_risk else 0
                hand_alert = self.hand_danger_frames >= self.alert_after_frames
                foot_alert = self.foot_danger_frames >= self.alert_after_frames
        else:
            self.hand_danger_frames = 0
            self.foot_danger_frames = 0
            hand_alert = False
            foot_alert = False

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
        if foot_alert:
            warnings.append("WARNING: FOOT NEAR MACHINE")
            self.log_alert("FOOT_NEAR_MACHINE", "Ban chan / co chan xam nhap vung may")
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
            for i, warning_text in enumerate(warnings):
                cv2.putText(
                    annotated, warning_text[:90], (10, 28 + i * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
                )

        return annotated

    def _publish_frame(self, annotated):
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            with self.frame_lock:
                self.latest_jpeg = buf.tobytes()

    def _run_loop(self):
        try:
            while not self.stop_event.is_set():
                if self.mode == "image":
                    if self.need_reprocess_image and self.raw_image is not None:
                        out = self._process_single_frame(self.raw_image.copy(), is_static_image=True)
                        self._publish_frame(out)
                        out_path = os.path.join(
                            "output", "out_" + os.path.basename(self.source_path)
                        )
                        ext = os.path.splitext(out_path)[1].lower()
                        ok, buf = cv2.imencode(ext, out)
                        if ok:
                            buf.tofile(out_path)
                        self.need_reprocess_image = False
                    time.sleep(0.1)

                elif self.mode == "video":
                    if self.video_cap is None:
                        break
                    ret, frame = self.video_cap.read()
                    if not ret or frame is None:
                        if self.video_writer is not None:
                            self.video_writer.release()
                            self.video_writer = None
                        self.mode = "idle"
                        self.source_label += " (Hoàn tất)"
                        break
                    out = self._process_single_frame(frame, is_static_image=False)
                    if self.video_writer is not None:
                        self.video_writer.write(out)
                    self._publish_frame(out)

                elif self.mode == "camera":
                    if self.camera is None:
                        break
                    result = self.camera.read()
                    if isinstance(result, tuple):
                        ret, frame = result
                        if not ret:
                            time.sleep(0.02)
                            continue
                    else:
                        frame = result

                    if frame is None:
                        self._render_placeholder("Mat ket noi Camera.")
                        time.sleep(0.05)
                        continue

                    out = self._process_single_frame(frame, is_static_image=False)
                    self._publish_frame(out)
                else:
                    time.sleep(0.05)
        except Exception as exc:
            print("[DETECTOR ERROR]", repr(exc))
            self._render_placeholder("LOI XU LY: " + str(exc)[:80])

    def get_jpeg_frame(self):
        with self.frame_lock:
            return self.latest_jpeg

    def get_status(self):
        with self.audio_lock:
            audio_active = (
                self.audio_alert_active
                and (time.monotonic() - self.audio_alert_time <= self.audio_alert_display_seconds)
            )
            audio_reason = self.audio_alert_reason

        with self.log_lock:
            logs = self.recent_logs[:15]

        return {
            "mode": self.mode,
            "source_label": self.source_label,
            "fps": round(self.fps_value, 1) if self.mode in ("camera", "video") else 0,
            "machine_zone": self.enable_machine_zone,
            "warnings": list(self.current_warnings),
            "logs": logs,
            "audio_active": audio_active,
            "audio_reason": audio_reason,
        }