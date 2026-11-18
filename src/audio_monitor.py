import json
import queue
import threading
import time
import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter
from vosk import Model, KaldiRecognizer


class AudioDistressMonitor:
    def __init__(
        self,
        model_path="models/vosk-vn",
        alert_callback=None,
        sample_rate=16000,
        cooldown=6,
        scream_multiplier=3.0,     # Tiếng hét = RMS nhiễu nền x 3.0
        min_scream_threshold=0.12,  # Ngưỡng RMS tối thiểu để coi là tiếng hét
        scream_duration=0.5,       # Tiếng hét kéo dài trên 0.5s sẽ báo động
        device=None,
        gain=2.5,                  # Hệ số khuếch đại âm thanh thu vào (Gain boost)
    ):
        self.model_path = model_path
        self.alert_callback = alert_callback
        self.sample_rate = sample_rate
        self.cooldown = cooldown
        self.scream_multiplier = scream_multiplier
        self.min_scream_threshold = min_scream_threshold
        self.scream_duration = scream_duration
        self.device = device
        self.gain = gain

        self.audio_queue = queue.Queue(maxsize=40)
        self.stop_event = threading.Event()
        self.last_alert_time = 0.0
        self.loud_duration = 0.0
        self.worker = None

        # Độ ồn nền khởi tạo (Adaptive Noise Floor)
        self.noise_floor = 0.02

        # Khởi tạo bộ lọc High-Pass Butterworth
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = 80.0 / nyquist
        self.b_hp, self.a_hp = butter(1, normal_cutoff, btype="high", analog=False)

        self.distress_phrases = [
            "cứu tôi",
            "cứu với",
            "cứu",
            "giúp tôi",
            "giúp với",
            "help me",
            "help",
            "đau quá",
            "bị kẹt",
            "ngã rồi",
        ]

    def _apply_highpass_filter(self, signal):
        """
        Bộ lọc High-Pass Butterworth sử dụng SciPy (tốc độ cao).
        Triệt tiêu các tần số thấp (< 80Hz) như tiếng ù máy móc, quạt thông gió.
        """
        try:
            return lfilter(self.b_hp, self.a_hp, signal)
        except Exception:
            return signal

    def _audio_callback(self, indata, frames, time_info, status):
        try:
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass

    def _raise_alert(self, reason):
        now = time.monotonic()
        if now - self.last_alert_time < self.cooldown:
            return

        self.last_alert_time = now

        print("\n" + "=" * 50)
        print("[CANH BAO AM THANH]", reason)
        print("=" * 50)

        if self.alert_callback:
            try:
                self.alert_callback(reason, 0.0)
            except Exception as exc:
                print("[AUDIO CALLBACK ERROR]", exc)

    def _process_audio(self):
        model = Model(self.model_path)

        # GIẢI PHÁP TỐI ƯU VOSK:
        # Ép mô hình chỉ tập trung nhận diện danh sách từ khóa + từ không xác định [unk]
        # Giúp tăng tốc độ xử lý và tăng độ chính xác lên gấp nhiều lần trong môi trường nhiễu.
        grammar_list = self.distress_phrases + ["[unk]"]
        grammar_json = json.dumps(grammar_list, ensure_ascii=False)
        recognizer = KaldiRecognizer(model, self.sample_rate, grammar_json)

        block_duration = 4000 / self.sample_rate
        print("[AUDIO] Micro dang hoat dong voi bo loc am thanh toi uu.")

        while not self.stop_event.is_set():
            try:
                audio = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            mono = audio[:, 0]

            # 1. Khai thác & Lọc âm thanh (Nhanh với SciPy)
            mono_filtered = self._apply_highpass_filter(mono)  # Lọc tiếng ù tần số thấp
            mono_boosted = mono_filtered * self.gain            # Khuếch đại âm thanh
            mono_clipped = np.clip(mono_boosted, -1.0, 1.0)     # Giới hạn tránh vỡ tiếng

            # 2. Xử lý Phát hiện Tiếng hét (RMS Adaptive)
            rms = float(np.sqrt(np.mean(mono_clipped ** 2)))

            # Cập nhật tiếng ồn nền động (Adaptive Noise Floor)
            if rms < self.min_scream_threshold:
                self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms

            # Ngưỡng hét linh hoạt theo môi trường thực tế
            dynamic_scream_threshold = max(
                self.min_scream_threshold,
                self.noise_floor * self.scream_multiplier
            )

            if rms >= dynamic_scream_threshold:
                self.loud_duration += block_duration
            else:
                self.loud_duration = max(0.0, self.loud_duration - block_duration)

            if self.loud_duration >= self.scream_duration:
                self._raise_alert("Phat hien tieng het hoac am thanh lon keo dai")
                self.loud_duration = 0.0

            # 3. Xử lý Nhận diện Từ khóa Lời nói (Speech-To-Text)
            pcm16 = (mono_clipped * 32767).astype(np.int16)

            recognized_text = ""

            # Thử nhận diện câu hoàn chỉnh hoặc từ khóa tức thì
            if recognizer.AcceptWaveform(pcm16.tobytes()):
                result = json.loads(recognizer.Result())
                recognized_text = result.get("text", "").lower().strip()
            else:
                partial_res = json.loads(recognizer.PartialResult())
                recognized_text = partial_res.get("partial", "").lower().strip()

            if recognized_text:
                # Bỏ từ nhiễu không xác định [unk]
                clean_text = recognized_text.replace("[unk]", "").strip()

                if clean_text:
                    for phrase in self.distress_phrases:
                        if phrase in clean_text:
                            self._raise_alert(f"Phat hien loi keu cuu: {phrase}")
                            recognizer.Reset()  # Reset bộ nhận diện để không lặp lại từ vừa bắt
                            break

    def _run(self):
        try:
            if self.stop_event.is_set():
                return

            def _create_stream(dev):
                kwargs = {
                    "samplerate": self.sample_rate,
                    "channels": 1,
                    "dtype": "float32",
                    "blocksize": 4000,
                    "callback": self._audio_callback,
                }
                if dev is not None:
                    kwargs["device"] = dev
                return sd.InputStream(**kwargs)

            stream = None
            # Thử với thiết bị yêu cầu, nếu thất bại sẽ dùng Micro mặc định
            for dev in (self.device, None):
                try:
                    stream = _create_stream(dev)
                    break
                except Exception:
                    continue

            if stream is None:
                raise RuntimeError("Khong tim thay microphone hop le.")

            with stream:
                self._process_audio()

        except Exception as exc:
            if not self.stop_event.is_set():
                print("[AUDIO ERROR]", exc)

    def start(self):
        if self.worker and self.worker.is_alive():
            return

        self.stop_event.clear()
        self.worker = threading.Thread(
            target=self._run,
            daemon=True,
        )
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=3)