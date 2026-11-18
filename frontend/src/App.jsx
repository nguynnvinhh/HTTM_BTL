import { useState, useEffect } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:5000";

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
  const [loading, setLoading] = useState(false);

  // Lay danh sach file co san trong thu muc input/ khi mo trang
  useEffect(() => {
    fetch(`${API_BASE}/api/files`)
      .then((res) => res.json())
      .then((data) => {
        if (data.ok && data.files) setInputFiles(data.files);
      })
      .catch(() => {});
  }, []);

  // Cap nhat trang thai canh bao & FPS moi 600ms
  useEffect(() => {
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/status`);
        const data = await res.json();
        setStatus(data);
      } catch {
        // Bo qua neu backend chua san sang
      }
    }, 600);

    return () => clearInterval(timer);
  }, []);

  const handleStartCamera = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/source/camera`, { method: "POST" });
      const data = await res.json();
      if (!data.ok) alert(data.message);
    } finally {
      setLoading(false);
    }
  };

  const handleStartExisting = async () => {
    if (!selectedFile) return alert("Vui lòng chọn một file trong danh sách.");
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/source/existing`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: selectedFile }),
      });
      const data = await res.json();
      if (!data.ok) alert(data.message);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadAndRun = async () => {
    if (!uploadFile) return alert("Vui lòng chọn file ảnh hoặc video từ máy.");
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", uploadFile);

      const res = await fetch(`${API_BASE}/api/source/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!data.ok) return alert(data.message);
      if (data.files) setInputFiles(data.files);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleMachineZone = async () => {
    await fetch(`${API_BASE}/api/machine_zone`, { method: "POST" });
  };

  const handleStop = async () => {
    await fetch(`${API_BASE}/api/stop`, { method: "POST" });
  };

  const hasWarning = status.warnings && status.warnings.length > 0;

  return (
    <div className="dashboard">
      <header className="topbar">
        <h1>Workplace Safety Detection</h1>
        <span className="fps-badge">FPS: {status.fps}</span>
      </header>

      <main className="main-grid">
        {/* Khu vuc hien thi Video / Anh */}
        <section className="video-card">
          <div className="video-header">
            <span>Nguồn: {status.source_label}</span>
            <span className={status.machine_zone ? "mz-on" : "mz-off"}>
              Machine Zone: {status.machine_zone ? "ON" : "OFF"}
            </span>
          </div>
          <div className="video-wrapper">
            <img src={`${API_BASE}/video_feed`} alt="Safety Detection Stream" />
          </div>
        </section>

        {/* Bang dieu khien ben phai */}
        <aside className="sidebar">
          <div className="card">
            <h2>Trạng thái cảnh báo</h2>
            <div className={`alert-box ${hasWarning ? "danger" : "safe"}`}>
              {hasWarning ? (
                status.warnings.map((w, idx) => <div key={idx}>{w}</div>)
              ) : (
                <div>An toàn - Không có cảnh báo</div>
              )}
            </div>
          </div>

          <div className="card">
            <h2>Nguồn đầu vào</h2>
            <button
              className="btn btn-primary"
              onClick={handleStartCamera}
              disabled={loading}
            >
              📷 Bật Camera trực tiếp
            </button>

            <select
              className="input-control"
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
            >
              <option value="">-- Chọn file có sẵn trong input/ --</option>
              {inputFiles.map((file) => (
                <option key={file} value={file}>
                  {file}
                </option>
              ))}
            </select>
            <button
              className="btn btn-secondary"
              onClick={handleStartExisting}
              disabled={loading}
            >
              ▶️ Chạy file đã chọn
            </button>

            <input
              type="file"
              className="input-control"
              accept=".jpg,.jpeg,.png,.bmp,.webp,.mp4,.avi,.mov,.mkv"
              onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
            />
            <button
              className="btn btn-secondary"
              onClick={handleUploadAndRun}
              disabled={loading}
            >
              ⬆️ Tải lên & Nhận diện
            </button>

            <hr className="divider" />

            <button className="btn btn-warning" onClick={handleToggleMachineZone}>
              ⚠️ Bật / Tắt Vùng Máy
            </button>
            <button className="btn btn-danger" onClick={handleStop}>
              ⏹️ Dừng giám sát
            </button>
          </div>

          <div className="card">
            <h2>Nhật ký cảnh báo</h2>
            <div className="log-list">
              {status.logs && status.logs.length > 0 ? (
                status.logs.map((item, idx) => (
                  <div className="log-item" key={idx}>
                    <div className="log-time">
                      [{item.time}] {item.type}
                    </div>
                    <div>{item.detail}</div>
                  </div>
                ))
              ) : (
                <div className="empty-log">Chưa có sự kiện nào được ghi nhận.</div>
              )}
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}