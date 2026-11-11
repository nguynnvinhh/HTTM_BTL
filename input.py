import cv2


class Camera:
    def __init__(self, camera_id=1):
        # Bắt buộc dùng DirectShow để đọc đúng luồng từ Iriun trên Windows
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise Exception(f"Không thể mở camera index {camera_id}!")

        # Khóa độ phân giải 640x480 để khung hình truyền tức thì, không bị đen do nghẽn buffer
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Đọc bỏ qua 2-3 frame rác khởi động đầu tiên của camera
        for _ in range(3):
            self.cap.read()

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()