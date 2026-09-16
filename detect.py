import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Cấu hình thư mục đầu vào và đầu ra
INPUT_DIR = "input"  # Thư mục chứa cả ảnh và video
OUTPUT_DIR = "output"  # Thư mục lưu kết quả

# 2. Nạp mô hình
MODEL_FALL = "best.pt"
POSE_BACKBONE = "yolov8s-pose.pt"

print("Đang nạp mô hình AI...")
fall_model = YOLO(MODEL_FALL)
pose_model = YOLO(POSE_BACKBONE)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_laying_by_geometry(keypoints, bbox):
    """Xác định tư thế nằm ngang qua tỷ lệ khung bao và trục vai-hông"""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    aspect_ratio = w / max(1, h)

    if keypoints is not None and len(keypoints) >= 17:
        kpts = keypoints.data[0].cpu().numpy()
        l_sh, r_sh = kpts[5], kpts[6]
        l_hip, r_hip = kpts[11], kpts[12]

        mid_sh = [(l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2]
        mid_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]

        dx = abs(mid_sh[0] - mid_hip[0])
        dy = abs(mid_sh[1] - mid_hip[1])

        # Trục cơ thể nghiêng hoặc song song mặt đất
        if dx > dy * 0.8:
            return True

    return aspect_ratio > 1.25


def process_frame(frame, imgsz=640):
    """Xử lý chung cho 1 khung hình (dùng cho cả ảnh tĩnh và frame video)"""
    # 1. Trích xuất Pose
    pose_results = pose_model(frame, imgsz=imgsz, conf=0.4, verbose=False)[0]
    annotated = pose_results.plot(boxes=False)

    # 2. Trích xuất nhãn fall từ best.pt
    fall_results = fall_model(frame, imgsz=imgsz, conf=0.35, verbose=False)[0]
    fall_boxes = []
    for f_box in fall_results.boxes:
        cls_name = fall_model.names[int(f_box.cls[0])]
        if cls_name == "laying":
            fall_boxes.append(f_box.xyxy[0].cpu().numpy())

    # 3. Phân loại từng đối tượng người
    for i, box in enumerate(pose_results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        kpts = pose_results.keypoints[i] if pose_results.keypoints is not None else None
        by_geom = is_laying_by_geometry(kpts, (x1, y1, x2, y2))

        by_model = False
        for f_b in fall_boxes:
            if not (x2 < f_b[0] or x1 > f_b[2] or y2 < f_b[1] or y1 > f_b[3]):
                by_model = True
                break

        is_laying = by_geom or by_model

        if is_laying:
            color = (0, 0, 255)  # Đỏ cảnh báo
            title = f"CANH BAO: TE NGA ({conf * 100:.1f}%)"
            thick = 3
        else:
            color = (0, 255, 0)  # Xanh bình thường
            title = f"Binh thuong ({conf * 100:.1f}%)"
            thick = 2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
        cv2.putText(
            annotated,
            title,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return annotated


def process_video_file(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Lỗi: Không mở được video {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    count = 0
    empty_limit = 0

    print(f"Bắt đầu render video (Tổng: ~{total_frames} frames)...")

    while cap.isOpened():
        ret, frame = cap.read()

        # Ngắt ngay khi không còn đọc được khung hình
        if not ret or frame is None:
            empty_limit += 1
            if empty_limit >= 2:
                break
            continue

        # Xử lý frame qua AI (giảm xuống 480 để xử lý nhanh nhất)
        processed_frame = process_frame(frame, imgsz=480)
        out.write(processed_frame)
        count += 1

        # In tiến trình bằng dòng mới chuẩn để không kẹt bộ đệm PyCharm
        if count % 25 == 0:
            print(f"-> Đã xử lý {count}/{total_frames} frames")
            sys.stdout.flush()

        # Điều kiện ngắt cứng khi vượt quá số lượng frame tổng
        if total_frames > 0 and count >= total_frames:
            break

    # Đóng video dứt điểm
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n[HOÀN TẤT 100%] Đã lưu video thành công: {output_path}")
    sys.stdout.flush()


# --- QUÉT VÀ XỬ LÝ TOÀN BỘ FOLDER ---
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".wmv")

if not os.path.exists(INPUT_DIR):
    print(f"Lỗi: Không tìm thấy thư mục '{INPUT_DIR}'")
    exit()

all_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(IMAGE_EXTS + VIDEO_EXTS)]

if not all_files:
    print(f"Không có file ảnh hoặc video hợp lệ nào trong '{INPUT_DIR}'")
    exit()

print(f"\n--- Tìm thấy {len(all_files)} files cần xử lý ---")

for idx, filename in enumerate(all_files, 1):
    in_file = os.path.join(INPUT_DIR, filename)

    if filename.lower().endswith(IMAGE_EXTS):
        out_file = os.path.join(OUTPUT_DIR, filename)
        print(f"\n[{idx}/{len(all_files)}] Đang xử lý ẢNH: {filename}")
        img = cv2.imread(in_file)
        if img is not None:
            res_img = process_frame(img, imgsz=960)
            cv2.imwrite(out_file, res_img)
            print(f"   -> [OK] Đã lưu ảnh: {out_file}")

    elif filename.lower().endswith(VIDEO_EXTS):
        base_name = os.path.splitext(filename)[0]
        out_file = os.path.join(OUTPUT_DIR, f"{base_name}_detected.mp4")
        print(f"\n[{idx}/{len(all_files)}] Đang xử lý VIDEO: {filename}")
        process_video_file(in_file, out_file)
        print(f"   -> [OK] Đã lưu video: {out_file}")

print(f"\n=== HOÀN TẤT TOÀN BỘ! Kết quả đã lưu tại thư mục: '{OUTPUT_DIR}' ===")