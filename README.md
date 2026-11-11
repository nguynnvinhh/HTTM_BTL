# Workplace Safety Detection

Hệ thống giám sát an toàn lao động theo thời gian thực sử dụng AI.  
Phát hiện: **ngã/nằm**, **tay gần máy nguy hiểm**, **tiếng kêu cứu**.

---

## Yêu cầu hệ thống

- Python **3.10** trở lên
- Windows (dùng DirectShow để đọc camera)
- Microphone (nếu dùng chế độ Camera)

---

## Cài đặt

### 1. Clone / tải project về

```bash
git clone <repo-url>
cd HTTM_BTL-main123
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Cài thư viện

```bash
pip install -r requirements.txt
```

> **Lưu ý GPU (tuỳ chọn):** Mặc định `torch` chạy trên CPU.  
> Nếu muốn tăng tốc bằng CUDA, cài `torch` riêng trước khi chạy lệnh trên:  
> → https://pytorch.org/get-started/locally/

### 4. Tải model giọng nói tiếng Việt (Vosk)

Tải model tại: https://alphacephei.com/vosk/models  
Chọn `vosk-model-vn-...`, giải nén, đặt vào:

```
models/
└── vosk-vn/
    ├── am/
    ├── conf/
    ├── graph/
    └── ivector/
```

> Model đã có sẵn trong project nếu clone từ repo đầy đủ.

---

## Cấu trúc thư mục

```
HTTM_BTL-main123/
├── detect.py           # File chạy chính
├── audio_monitor.py    # Module giám sát âm thanh
├── input.py            # Wrapper đọc camera
├── requirements.txt    # Danh sách thư viện
├── best.pt             # Model YOLO phát hiện ngã (custom)
├── yolov8s-pose.pt     # Model YOLOv8 Pose
├── input/              # Đặt video/ảnh cần xử lý vào đây
├── output/             # Kết quả và log cảnh báo lưu ở đây
└── models/
    └── vosk-vn/        # Model nhận dạng giọng nói tiếng Việt
```

---

## Cách chạy

```bash
python detect.py
```

Khi khởi động, hộp thoại sẽ hiện lên để chọn nguồn đầu vào:

| Lựa chọn | Mô tả |
|---|---|
| 📷 Camera trực tiếp | Dùng webcam (mặc định index 2) |
| File video/ảnh | Chọn file có sẵn trong `input/` hoặc thư mục gốc |
| Browse... | Duyệt chọn file bất kỳ trên máy |
| ⚠️ Bật vùng máy | Tùy chọn bật/tắt Machine Zone (tự động tắt khi chọn video/ảnh để không bị vướng) |

### Phím tắt trong lúc đang chạy:
- **`Q`**: Thoát chương trình
- **`M`**: Bật / Tắt nhanh khung **Machine Zone** và tính năng cảnh báo tay gần máy tức thì.

---

## Tính năng

- 🤸 **Phát hiện ngã / nằm** — kết hợp model AI + phân tích hình học (tỉ lệ bounding box, góc vai-hông)
- ✋ **Phát hiện tay gần máy** — theo dõi keypoint cổ tay trong vùng Machine Zone
- 🔊 **Giám sát âm thanh** — phát hiện tiếng la hét (RMS) và nhận dạng lời kêu cứu (Vosk, tiếng Việt)
- 📊 **FPS counter** — hiển thị tốc độ xử lý góc trên phải
- 🟠 **Machine Zone** — vẽ vùng nguy hiểm lên màn hình để kiểm tra trực quan
- 📝 **Log cảnh báo** — tự động lưu vào `output/alert_log_<timestamp>.csv`
- 🎞️ **Hỗ trợ video/ảnh** — xử lý và lưu kết quả vào `output/`

---

## Thư viện sử dụng

| Thư viện | Mục đích |
|---|---|
| `ultralytics` | YOLOv8 — phát hiện người, tư thế, ngã |
| `opencv-python` | Đọc camera/video/ảnh, vẽ chú thích |
| `torch` | Backend AI (cài tự động theo ultralytics) |
| `vosk` | Nhận dạng giọng nói offline tiếng Việt |
| `sounddevice` | Thu âm thanh từ microphone |
| `numpy` | Xử lý tín hiệu âm thanh, keypoints |
| `tkinter` | Hộp thoại chọn nguồn (built-in Python) |
