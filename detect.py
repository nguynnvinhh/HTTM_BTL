
import os
import csv
import cv2
import time
import threading
from datetime import datetime
from tkinter import (
    Tk, StringVar, Label, Button, Radiobutton,
    Frame, Scrollbar, Listbox, SINGLE, END, BOTH, LEFT, RIGHT, Y, X,
    messagebox, filedialog,
)

from ultralytics import YOLO
from input import Camera
from audio_monitor import AudioDistressMonitor


# =====================================================
# 0. HOP THOAI CHON NGUON DAU VAO
# =====================================================

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEDIA_EXTS  = VIDEO_EXTS | IMAGE_EXTS


def _scan_media_files():
    """Tim tat ca file anh/video trong thu muc 'input/' va thu muc goc project."""
    found = []
    search_dirs = [
        os.path.abspath("input"),
        os.path.abspath("."),
    ]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            ext = os.path.splitext(name)[1].lower()
            if ext in MEDIA_EXTS:
                full = os.path.join(d, name)
                if full not in found:
                    found.append(full)
    return found


def show_source_selector():
    """
    Mo hop thoai tkinter de nguoi dung chon nguon dau vao.
    Tra ve:
        None          -> dung camera
        str (path)    -> duong dan toi file video hoac anh
        'CANCEL'      -> nguoi dung dong cua so, thoat chuong trinh
    """
    root = Tk()
    root.title("Chon nguon dau vao")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    chosen = {"value": "CANCEL"}

    # --- Tieu de ---
    Label(
        root,
        text="Workplace Safety Detection",
        font=("Helvetica", 13, "bold"),
        pady=6,
    ).pack()

    Label(
        root,
        text="Chon nguon dau vao:",
        font=("Helvetica", 10),
        anchor="w",
        padx=16,
    ).pack(fill=X)

    # --- Radio: Camera ---
    sel_var = StringVar(value="__camera__")

    cam_frame = Frame(root, padx=16, pady=4)
    cam_frame.pack(fill=X)
    Radiobutton(
        cam_frame,
        text="📷  Camera truc tiep  (index 2)",
        variable=sel_var,
        value="__camera__",
        font=("Helvetica", 10),
    ).pack(anchor="w")

    # --- Danh sach file co san ---
    files = _scan_media_files()

    list_frame = Frame(root, padx=16)
    list_frame.pack(fill=BOTH, expand=True)

    Label(
        list_frame,
        text="Hoac chon file video / anh co san:",
        font=("Helvetica", 9),
        anchor="w",
    ).pack(anchor="w", pady=(6, 2))

    scroll = Scrollbar(list_frame, orient="vertical")
    lb = Listbox(
        list_frame,
        selectmode=SINGLE,
        yscrollcommand=scroll.set,
        height=min(max(len(files), 3), 10),
        font=("Consolas", 9),
        activestyle="dotbox",
    )
    scroll.config(command=lb.yview)
    scroll.pack(side=RIGHT, fill=Y)
    lb.pack(side=LEFT, fill=BOTH, expand=True)

    for f in files:
        lb.insert(END, os.path.relpath(f))

    def on_listbox_select(event):
        idxs = lb.curselection()
        if idxs:
            sel_var.set(files[idxs[0]])

    lb.bind("<<ListboxSelect>>", on_listbox_select)

    # --- Nut Browse ---
    def on_browse():
        path = filedialog.askopenfilename(
            title="Chon file video hoac anh",
            filetypes=[
                ("Video / Anh", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp"),
                ("Tat ca", "*.*"),
            ],
        )
        if path:
            sel_var.set(path)
            # Them vao listbox neu chua co
            rel = os.path.relpath(path)
            existing = lb.get(0, END)
            if rel not in existing:
                lb.insert(END, rel)
                files.append(path)
            # Chon dong vua them
            for i, item in enumerate(lb.get(0, END)):
                if item == rel:
                    lb.selection_clear(0, END)
                    lb.selection_set(i)
                    lb.see(i)
                    break

    browse_btn = Button(
        root,
        text="🗂  Browse file khac...",
        command=on_browse,
        font=("Helvetica", 9),
        padx=8,
    )
    browse_btn.pack(pady=(4, 0))

    # --- Nut OK / Huy ---
    btn_frame = Frame(root, pady=10)
    btn_frame.pack()

    def on_ok():
        v = sel_var.get()
        if v == "__camera__":
            chosen["value"] = None
        else:
            if not os.path.isfile(v):
                messagebox.showerror("Loi", f"Khong tim thay file:\n{v}")
                return
            chosen["value"] = v
        root.destroy()

    def on_cancel():
        chosen["value"] = "CANCEL"
        root.destroy()

    Button(
        btn_frame,
        text="  OK  ",
        command=on_ok,
        font=("Helvetica", 10, "bold"),
        bg="#2196F3",
        fg="white",
        padx=14,
        pady=4,
    ).pack(side=LEFT, padx=8)

    Button(
        btn_frame,
        text="  Huy  ",
        command=on_cancel,
        font=("Helvetica", 10),
        padx=14,
        pady=4,
    ).pack(side=LEFT)

    # Can giua man hinh
    root.update_idletasks()
    w = root.winfo_width()
    h = root.winfo_height()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()

    return chosen["value"]


SOURCE_SELECTION = show_source_selector()
if SOURCE_SELECTION == "CANCEL":
    raise SystemExit("Nguoi dung huy chon nguon dau vao.")



# =====================================================
# 1. CAU HINH
# =====================================================

MODEL_FALL = "best.pt"
POSE_BACKBONE = "yolov8s-pose.pt"

CAMERA_ID = 2
WINDOW_NAME = "Workplace Safety Detection"

POSE_CONF = 0.30
FALL_CONF = 0.30
WRIST_CONF = 0.25

# So frame lien tiep co co tay gan may de canh bao
ALERT_AFTER_FRAMES = 2

# Hien thi canh bao am thanh trong so giay nay
AUDIO_ALERT_DISPLAY_SECONDS = 6

# File log canh bao
LOG_FILE = os.path.join(
    "output",
    "alert_log_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
)

danger_frames = 0

# Trang thai canh bao am thanh, dung chung giua cac luong
audio_alert_active = False
audio_alert_reason = ""
audio_alert_time = 0.0

audio_alert_lock = threading.Lock()

# FPS tracking
_fps_prev_time = time.monotonic()
_fps_value = 0.0


# =====================================================
# 1b. KHOI TAO LOG FILE
# =====================================================

os.makedirs("output", exist_ok=True)

with open(LOG_FILE, "w", newline="", encoding="utf-8") as _f:
    csv.writer(_f).writerow(["timestamp", "alert_type", "detail"])


def log_alert(alert_type: str, detail: str = ""):
    """Ghi mot dong canh bao vao file CSV."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([timestamp, alert_type, detail])
    except Exception as exc:
        print("[LOG ERROR]", exc)


# =====================================================
# 2. KIEM TRA VA NAP MO HINH
# =====================================================

for path in (MODEL_FALL, POSE_BACKBONE):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Khong tim thay mo hinh: {path}"
        )

fall_model = YOLO(MODEL_FALL)
pose_model = YOLO(POSE_BACKBONE)

print("Fall model:", fall_model.names)
print("Pose model:", pose_model.names)


# =====================================================
# 3. VUNG MAY
# =====================================================

def get_machine_region(frame):
    """
    Vung may tham khao theo ti le khung hinh.
    Can dieu chinh theo camera thuc te.
    """
    h, w = frame.shape[:2]

    x1 = int(0.25 * w)
    y1 = int(0.30 * h)
    x2 = int(0.98 * w)
    y2 = int(0.68 * h)

    return x1, y1, x2, y2


def draw_machine_region(annotated, frame):
    """Ve vung may len khung hinh de nguoi dung thay duoc."""
    x1, y1, x2, y2 = get_machine_region(frame)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
    cv2.putText(
        annotated,
        "Machine Zone",
        (x1 + 4, y1 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 165, 255),
        1,
    )


# =====================================================
# 4. PHAT HIEN CO TAY GAN MAY
# =====================================================

def detect_hand_near_machine(frame, pose_result, annotated):
    x1, y1, x2, y2 = get_machine_region(frame)

    risk = False

    if pose_result.keypoints is None:
        return False

    for person_idx in range(len(pose_result.keypoints)):
        points = (
            pose_result.keypoints[person_idx]
            .data[0]
            .cpu()
            .numpy()
        )

        # COCO Pose: 9 = co tay trai, 10 = co tay phai
        for wrist_idx in (9, 10):
            px_float, py_float, confidence = points[wrist_idx][:3]

            if confidence < WRIST_CONF:
                continue

            px = int(px_float)
            py = int(py_float)

            inside_region = (
                x1 <= px <= x2
                and y1 <= py <= y2
            )

            if inside_region:
                risk = True

                cv2.circle(
                    annotated,
                    (px, py),
                    12,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    annotated,
                    "WRIST NEAR MACHINE",
                    (max(5, px - 150), max(25, py - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

    return risk


# =====================================================
# 5. PHAT HIEN TU THE NAM / NGA
# =====================================================

def is_laying_by_geometry(keypoints, bbox):
    x1, y1, x2, y2 = bbox

    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)

    aspect_ratio = box_width / box_height

    if keypoints is not None:
        try:
            kpts = keypoints.data[0].cpu().numpy()
        except (AttributeError, IndexError, TypeError):
            kpts = None

        if kpts is not None and len(kpts) >= 17:
            left_hip = kpts[11]
            right_hip = kpts[12]
            left_shoulder = kpts[5]
            right_shoulder = kpts[6]

            hip_conf = (
                left_hip[2] + right_hip[2]
            ) / 2

            shoulder_conf = (
                left_shoulder[2] + right_shoulder[2]
            ) / 2

            if hip_conf > 0.35 and shoulder_conf > 0.35:
                shoulder_x = (
                    left_shoulder[0] + right_shoulder[0]
                ) / 2

                shoulder_y = (
                    left_shoulder[1] + right_shoulder[1]
                ) / 2

                hip_x = (
                    left_hip[0] + right_hip[0]
                ) / 2

                hip_y = (
                    left_hip[1] + right_hip[1]
                ) / 2

                dx = abs(shoulder_x - hip_x)
                dy = abs(shoulder_y - hip_y)

                if dx > dy * 1.2:
                    return True

    return aspect_ratio > 1.45


# =====================================================
# 6. CALLBACK CANH BAO AM THANH
# =====================================================

def on_audio_alert(reason, confidence=0.0):
    """
    Duoc goi tu luong am thanh.
    Chi cap nhat trang thai, khong xu ly YOLO tai day.
    """
    global audio_alert_active
    global audio_alert_reason
    global audio_alert_time

    with audio_alert_lock:
        audio_alert_active = True
        audio_alert_reason = str(reason)
        audio_alert_time = time.monotonic()

    log_alert("AUDIO_DISTRESS_RAW", str(reason))

    print("\n" + "=" * 55)
    print("CANH BAO AM THANH: CO THE CO NGUOI KEU CUU!")
    print("Ly do:", reason)
    print("=" * 55)


# =====================================================
# 7. XU LY MOT FRAME
# =====================================================

def process_frame(frame):
    global danger_frames, _fps_prev_time, _fps_value

    # ----- FPS -----
    now = time.monotonic()
    _fps_value = 1.0 / max(now - _fps_prev_time, 1e-6)
    _fps_prev_time = now

    # ----- Pose -----
    pose_result = pose_model.predict(
        source=frame,
        imgsz=640,
        conf=POSE_CONF,
        verbose=False
    )[0]

    annotated = pose_result.plot(boxes=False)

    # ----- Ve vung may len man hinh -----
    draw_machine_region(annotated, frame)

    # ----- Model phat hien nga -----
    fall_result = fall_model.predict(
        source=frame,
        imgsz=640,
        conf=FALL_CONF,
        verbose=False
    )[0]

    fall_boxes = []

    if fall_result.boxes is not None:
        for box in fall_result.boxes:
            class_id = int(box.cls[0].item())
            names = fall_model.names

            if isinstance(names, dict):
                class_name = str(
                    names.get(class_id, class_id)
                ).lower().strip()
            else:
                class_name = str(
                    names[class_id]
                ).lower().strip()

            if class_name in {
                "laying", "fall", "fallen", "falling"
            }:
                fall_boxes.append(
                    tuple(
                        int(v)
                        for v in box.xyxy[0].cpu().tolist()
                    )
                )

    # ----- Ve nguoi va phat hien nga -----
    fall_detected = False

    if pose_result.boxes is not None:
        for index, box in enumerate(pose_result.boxes):
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].cpu().tolist()
            )

            confidence = float(box.conf[0].item())

            keypoints = (
                pose_result.keypoints[index]
                if pose_result.keypoints is not None
                else None
            )

            by_geometry = is_laying_by_geometry(
                keypoints,
                (x1, y1, x2, y2)
            )

            by_model = any(
                not (
                    x2 < fx1 or x1 > fx2
                    or y2 < fy1 or y1 > fy2
                )
                for fx1, fy1, fx2, fy2 in fall_boxes
            )

            if by_geometry or by_model:
                fall_detected = True
                color = (0, 0, 255)
                label = "WARNING: FALL / LYING"
            else:
                color = (0, 255, 0)
                label = f"Person ({confidence:.2f})"

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                annotated,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # ----- Canh bao co tay gan may -----
    hand_risk = detect_hand_near_machine(
        frame,
        pose_result,
        annotated
    )

    if hand_risk:
        danger_frames += 1
    else:
        danger_frames = 0

    hand_alert = danger_frames >= ALERT_AFTER_FRAMES

    # ----- Lay trang thai canh bao am thanh mot cach an toan -----
    with audio_alert_lock:
        sound_active = (
            audio_alert_active
            and time.monotonic() - audio_alert_time
            <= AUDIO_ALERT_DISPLAY_SECONDS
        )
        sound_reason = audio_alert_reason

    # ----- Tong hop canh bao -----
    warnings = []

    if hand_alert:
        warnings.append("WARNING: HAND NEAR MACHINE")
        log_alert("HAND_NEAR_MACHINE")

    if fall_detected:
        warnings.append("WARNING: FALL / LYING")
        log_alert("FALL_DETECTED")

    if sound_active:
        warnings.append("CANH BAO: CO NGUOI KEU CUU!")
        warnings.append("Am thanh: " + sound_reason)
        log_alert("AUDIO_DISTRESS", sound_reason)

    # ----- Hien thi FPS goc tren phai -----
    fps_text = f"FPS: {_fps_value:.1f}"
    fps_x = max(0, annotated.shape[1] - 130)
    cv2.putText(
        annotated,
        fps_text,
        (fps_x, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 180),
        2,
    )

    # ----- Hien thi banner tren camera -----
    if warnings:
        banner_height = min(
            35 * len(warnings) + 12,
            annotated.shape[0] - 1
        )
        banner_width = min(
            annotated.shape[1] - 1,
            1100
        )

        # Mau do cho canh bao
        cv2.rectangle(
            annotated,
            (0, 0),
            (banner_width, banner_height),
            (0, 0, 220),
            -1
        )

        for i, warning in enumerate(warnings):
            cv2.putText(
                annotated,
                warning[:100],
                (10, 28 + i * 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

    return annotated


def _draw_source_label(annotated, label: str):
    """Hien thi ten nguon dau vao goc duoi trai."""
    h = annotated.shape[0]
    cv2.putText(
        annotated,
        label,
        (8, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


# =====================================================
# 8. CHAY CAMERA / VIDEO / ANH
# =====================================================

SOURCE = SOURCE_SELECTION  # None → camera, str path → video/anh

# Phan loai nguon dau vao
_src_ext = os.path.splitext(SOURCE or "")[1].lower()
IS_IMAGE = SOURCE is not None and _src_ext in IMAGE_EXTS
IS_VIDEO = SOURCE is not None and not IS_IMAGE

camera = None
audio_monitor = None

try:
    # ------- MO NGUON DAU VAO -------
    if IS_IMAGE:
        # Anh don: xu ly 1 lan, hien thi va luu
        print(f"[INFO] Che do anh: {SOURCE}")
        frame = cv2.imread(SOURCE)
        if frame is None:
            raise FileNotFoundError(f"Khong mo duoc anh: {SOURCE}")

        output = process_frame(frame)
        _draw_source_label(output, f"[IMAGE] {os.path.basename(SOURCE)}")

        out_name = "out_" + os.path.basename(SOURCE)
        out_path = os.path.join("output", out_name)
        cv2.imwrite(out_path, output)
        print(f"[INFO] Da luu ket qua: {out_path}")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, output)
        print("Nhan phim bat ky de dong...")
        cv2.waitKey(0)

    elif IS_VIDEO:
        # Video file
        print(f"[INFO] Che do video: {SOURCE}")
        cap = cv2.VideoCapture(SOURCE)
        if not cap.isOpened():
            raise FileNotFoundError(f"Khong mo duoc video: {SOURCE}")

        # Ghi video ket qua
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_name = "out_" + os.path.basename(SOURCE)
        out_path = os.path.join("output", out_name)
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_in,
            (w, h),
        )

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
        print("Nhan Q de thoat som.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            output = process_frame(frame)
            _draw_source_label(output, f"[VIDEO] {os.path.basename(SOURCE)}")
            writer.write(output)
            cv2.imshow(WINDOW_NAME, output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        writer.release()
        print(f"[INFO] Da luu video ket qua: {out_path}")

    else:
        # Camera truc tiep (mac dinh)
        print("Dang khoi dong camera...")

        camera = Camera(CAMERA_ID)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

        # Khoi dong am thanh doc lap voi camera
        audio_monitor = AudioDistressMonitor(
            model_path="models/vosk-vn",
            alert_callback=on_audio_alert
        )

        audio_monitor.start()

        print("Camera va audio dang khoi dong.")
        print("Nhan Q de thoat.")

        while True:
            frame = camera.read()

            if frame is None:
                print("Khong doc duoc frame.")
                break

            output = process_frame(frame)
            _draw_source_label(output, f"[CAM] index {CAMERA_ID}")

            cv2.imshow(WINDOW_NAME, output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except KeyboardInterrupt:
    print("Da dung chuong trinh.")

except Exception as exc:
    print("LOI:", exc)

finally:
    if audio_monitor is not None:
        audio_monitor.stop()

    if camera is not None:
        camera.release()

    cv2.destroyAllWindows()
    print("Da dong chuong trinh.")
