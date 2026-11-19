import { useState, useEffect, useCallback } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:5000";

const DEFAULT_CAMERAS = [
  { id: 0, label: "Camera Laptop (Index 0)" },
  { id: 1, label: "Iriun / Cam ngoài (Index 1)" },
  { id: 2, label: "Camera phụ (Index 2)" },
  { id: 3, label: "Camera phụ (Index 3)" },
];

export default function App() {
  const [status, setStatus] = useState({
    mode: "idle",
    source_label: "Chưa chọn nguồn",
    fps: 0,
    machine_zone: true,
    warnings: [],
    logs: [],
  });

  const [inputFiles, setInputFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState("");
  const [uploadFile, setUploadFile] = useState(null);
  const [cameras, setCameras] = useState(DEFAULT_CAMERAS);
  const [cameraIndex, setCameraIndex] = useState("0");
  const [loading, setLoading] = useState(false);
  const [backendConnected, setBackendConnected] = useState(false);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/status`);
      if (!response.ok) throw new Error("Backend trả về lỗi.");
      const data = await response.json();
      setStatus((previous) => ({ ...previous, ...data }));
      setBackendConnected(true);
    } catch {
      setBackendConnected(false);
    }
  }, []);

  const fetchFiles = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/files`);
      const data = await response.json();
      if (data.ok && Array.isArray(data.files)) {
        setInputFiles(data.files);
      }
    } catch (error) {
      console.error("Không thể lấy danh sách file:", error);
    }
  }, []);

  const fetchCameras = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/cameras`);
      const data = await response.json();
      if (data.ok && Array.isArray(data.cameras) && data.cameras.length > 0) {
        setCameras(data.cameras);
        if (!data.cameras.some((c) => String(c.id) === String(cameraIndex))) {
          setCameraIndex(String(data.cameras[0].id));
        }
      }
    } catch (error) {
      console.error("Không thể quét danh sách camera:", error);
    }
  }, [cameraIndex]);

  useEffect(() => {
    fetchStatus();
    fetchFiles();
    fetchCameras();

    const timer = setInterval(() => {
      fetchStatus();
    }, 1000);

    return () => clearInterval(timer);
  }, [fetchStatus, fetchFiles, fetchCameras]);

  const handleStartCamera = async () => {
    const index = Number(cameraIndex);
    if (!Number.isInteger(index) || index < 0) {
      alert("Camera Index không hợp lệ.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/source/camera`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_index: index }),
      });
      const data = await response.json();

      if (!response.ok || !data.ok) {
        alert(data.message || "Không thể kết nối Camera.");
        return;
      }
      await fetchStatus();
    } catch {
      alert("Không thể kết nối Backend Flask. Hãy kiểm tra python app.py.");
    } finally {
      setLoading(false);
    }
  };

  const handleStartExisting = async () => {
    if (!selectedFile) {
      alert("Vui lòng chọn ảnh hoặc video.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/source/existing`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: selectedFile }),
      });
      const data = await response.json();

      if (!response.ok || !data.ok) {
        alert(data.message || "Không thể chạy file.");
        return;
      }
      await fetchStatus();
    } catch {
      alert("Không thể kết nối Backend Flask.");
    } finally {
      setLoading(false);
    }
  };

  const handleUploadAndRun = async () => {
    if (!uploadFile) {
      alert("Vui lòng chọn ảnh hoặc video cần nhận diện.");
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", uploadFile);

      const response = await fetch(`${API_BASE}/api/source/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok || !data.ok) {
        alert(data.message || "Upload thất bại.");
        return;
      }

      if (Array.isArray(data.files)) {
        setInputFiles(data.files);
      }

      setUploadFile(null);
      const fileInput = document.getElementById("upload-file");
      if (fileInput) fileInput.value = "";

      await fetchStatus();
    } catch {
      alert("Không thể tải file lên Backend.");
    } finally {
      setLoading(false);
    }
  };

  const handleToggleMachineZone = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/machine_zone`, {
        method: "POST",
      });
      const data = await response.json();
      if (!response.ok || !data.ok) {
        alert(data.message || "Không thể thay đổi vùng máy.");
        return;
      }
      await fetchStatus();
    } catch {
      alert("Không thể kết nối Backend.");
    }
  };

  const handleStop = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/stop`, {
        method: "POST",
      });
      const data = await response.json();
      if (!response.ok || !data.ok) {
        alert(data.message || "Không thể dừng giám sát.");
        return;
      }
      await fetchStatus();
    } catch {
      alert("Không thể gửi lệnh dừng đến Backend.");
    } finally {
      setLoading(false);
    }
  };

  const warnings = Array.isArray(status.warnings) ? status.warnings : [];
  const logs = Array.isArray(status.logs) ? status.logs : [];
  const hasWarning = warnings.length > 0;
  const isRunning = status.mode && status.mode !== "idle" && status.mode !== "error";

  return (
    <div className="dashboard">
      <header className="topbar">
        <div className="brand">
          <h1>WORKPLACE SAFETY DETECTION</h1>
          <p>Hệ thống giám sát an toàn lao động bằng YOLO & Vosk</p>
        </div>

        <div className="header-status">
          <span
            className={
              backendConnected
                ? "status-indicator online"
                : "status-indicator offline"
            }
          >
            {backendConnected ? "BACKEND ONLINE" : "BACKEND OFFLINE"}
          </span>
          <span className="fps-badge">FPS: {status.fps ?? 0}</span>
        </div>
      </header>

      <main className="main-grid">
        <section className="video-card">
          <div className="video-header">
            <div>
              <strong>CAMERA GIÁM SÁT</strong>
              <p>Nguồn: {status.source_label || "Chưa chọn nguồn"}</p>
            </div>
            <span className={status.machine_zone ? "mz-on" : "mz-off"}>
              Vùng máy: {status.machine_zone ? "ON" : "OFF"}
            </span>
          </div>

          <div className="video-wrapper">
            {backendConnected ? (
              <img
                src={`${API_BASE}/video_feed`}
                alt="Video nhận diện an toàn lao động"
              />
            ) : (
              <div className="video-placeholder">
                <h2>Không thể kết nối Backend</h2>
                <p>Hãy chạy python app.py</p>
              </div>
            )}
          </div>

          <div className="video-footer">
            <span>Trạng thái: {isRunning ? "Đang giám sát" : "Đang chờ"}</span>
            <span>FPS: {status.fps ?? 0}</span>
          </div>
        </section>

        <aside className="sidebar">
          <section className="card">
            <h2>Trạng thái cảnh báo</h2>
            <div className={hasWarning ? "alert-box danger" : "alert-box safe"}>
              {hasWarning ? (
                warnings.map((warning, index) => (
                  <div key={index} className="warning-item">
                    ⚠️{" "}
                    {typeof warning === "string"
                      ? warning
                      : JSON.stringify(warning)}
                  </div>
                ))
              ) : (
                <div>✓ An toàn - Không có cảnh báo</div>
              )}
            </div>
          </section>

          <section className="card">
            <h2>Nguồn Camera trực tiếp</h2>
            <p className="camera-hint">
              Chọn <strong>Index 0</strong> cho webcam tích hợp của laptop, hoặc{" "}
              <strong>Index 1</strong> cho điện thoại qua Iriun Webcam.
            </p>

            <label className="camera-help" htmlFor="camera-index">
              Thiết bị Camera
            </label>

            <select
              id="camera-index"
              className="input-control"
              value={cameraIndex}
              onChange={(event) => setCameraIndex(event.target.value)}
            >
              {cameras.map((cam) => (
                <option key={cam.id} value={String(cam.id)}>
                  {cam.label}
                </option>
              ))}
            </select>

            <button
              className="btn btn-primary"
              onClick={handleStartCamera}
              disabled={loading || !backendConnected}
            >
              {loading ? "Đang kết nối..." : "📷 Bật Camera đã chọn"}
            </button>

            <button
              className="btn btn-secondary"
              onClick={fetchCameras}
              disabled={loading || !backendConnected}
            >
              ↻ Quét lại thiết bị Camera
            </button>
          </section>

          <section className="card">
            <h2>Nguồn ảnh / video</h2>

            <label className="camera-help">
              Chọn file mẫu trong thư mục input
            </label>

            <select
              className="input-control"
              value={selectedFile}
              onChange={(event) => setSelectedFile(event.target.value)}
            >
              <option value="">-- Chọn file --</option>
              {inputFiles.map((file) => (
                <option key={file} value={file}>
                  {file}
                </option>
              ))}
            </select>

            <button
              className="btn btn-secondary"
              onClick={handleStartExisting}
              disabled={loading || !backendConnected}
            >
              ▶ Chạy file đã chọn
            </button>

            <button
              className="btn btn-secondary"
              onClick={fetchFiles}
              disabled={!backendConnected}
            >
              ↻ Làm mới danh sách
            </button>

            <div className="file-upload">
              <label className="camera-help" htmlFor="upload-file">
                Tải ảnh hoặc video từ máy tính
              </label>

              <input
                id="upload-file"
                type="file"
                className="input-control"
                accept=".jpg,.jpeg,.png,.bmp,.webp,.mp4,.avi,.mov,.mkv,.wmv,.webm"
                onChange={(event) =>
                  setUploadFile(event.target.files?.[0] || null)
                }
              />

              {uploadFile && (
                <p className="camera-hint">Đã chọn: {uploadFile.name}</p>
              )}

              <button
                className="btn btn-secondary"
                onClick={handleUploadAndRun}
                disabled={loading || !backendConnected}
              >
                ⬆ Tải lên và nhận diện
              </button>
            </div>
          </section>

          <section className="card">
            <h2>Điều khiển hệ thống</h2>

            <button
              className="btn btn-warning"
              onClick={handleToggleMachineZone}
              disabled={!backendConnected || loading}
            >
              ⚠️ Bật / Tắt vùng nguy hiểm máy
            </button>

            <button
              className="btn btn-danger"
              onClick={handleStop}
              disabled={!backendConnected || loading}
            >
              ⏹ Dừng giám sát
            </button>
          </section>

          <section className="card">
            <h2>Nhật ký cảnh báo</h2>
            <div className="log-list">
              {logs.length > 0 ? (
                logs.map((item, index) => (
                  <div className="log-item" key={index}>
                    <div className="log-time">
                      {item.time || "--:--:--"} {item.type || "Sự kiện"}
                    </div>
                    <div>{item.detail || item.message || ""}</div>
                  </div>
                ))
              ) : (
                <div className="empty-log">
                  Chưa có sự kiện nào được ghi nhận.
                </div>
              )}
            </div>
          </section>
        </aside>
      </main>
    </div>
  );
}