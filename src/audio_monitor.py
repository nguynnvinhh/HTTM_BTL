import json
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter
from vosk import Model, KaldiRecognizer


class AudioDistressMonitor:
    """
    Giám sát âm thanh kêu cứu & tiếng hét thời gian thực từ nhiều microphone.
    Hỗ trợ chạy đồng thời cả micro Iriun và cả micro Laptop,
    tự động bỏ qua thiết bị không khả dụng mà không làm gián đoạn luồng còn lại.
    """

    def __init__(
        self,
        model_path="models/vosk-vn",
        alert_callback=None,
        sample_rate=16000,
        cooldown=6.0,
        scream_multiplier=3.0,
        min_scream_threshold=0.12,
        scream_duration=0.5,
        device=None,
        gain=2.5,
        blocksize=4000,
    ):
        self.model_path = model_path
        self.alert_callback = alert_callback
        self.sample_rate = sample_rate
        self.cooldown = cooldown
        self.scream_multiplier = scream_multiplier
        self.min_scream_threshold = min_scream_threshold
        self.scream_duration = scream_duration
        self.gain = gain
        self.blocksize = blocksize

        # Hỗ trợ truyền 1 index (int), danh sách index ([4, 3]) hoặc None (mặc định)
        if device is None:
            self.devices = [None]
        elif isinstance(device, (list, tuple, set)):
            # Loại bỏ trùng lặp nhưng giữ nguyên thứ tự ưu tiên
            seen = set()
            self.devices = [d for d in device if not (d in seen or seen.add(d))]
        else:
            self.devices = [device]

        self.stop_event = threading.Event()
        self.worker = None
        self.streams = []
        self.processing_threads = []
        self.states = {}

        # Khóa chống báo động trùng giữa 2 microphone
        self.alert_lock = threading.Lock()
        self.last_alert_time = 0.0

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

        # Bộ lọc High-pass 80 Hz loại bỏ tiếng ù nền (quạt, động cơ)
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = 80.0 / nyquist
        self.b_hp, self.a_hp = butter(1, normal_cutoff, btype="high", analog=False)

    def _create_device_state(self, device, name):
        return {
            "device": device,
            "name": name,
            "audio_queue": queue.Queue(maxsize=30),
            "noise_floor": 0.02,
            "loud_duration": 0.0,
        }

    def _audio_callback(self, source_id):
        def callback(indata, frames, time_info, status):
            if self.stop_event.is_set():
                return

            audio_queue = self.states[source_id]["audio_queue"]
            chunk = indata.copy()
            try:
                audio_queue.put_nowait(chunk)
            except queue.Full:
                # Đẩy gói cũ nhất ra để ưu tiên dữ liệu âm thanh thời gian thực
                try:
                    audio_queue.get_nowait()
                    audio_queue.put_nowait(chunk)
                except (queue.Empty, queue.Full):
                    pass

        return callback

    def _apply_highpass_filter(self, signal):
        try:
            return lfilter(self.b_hp, self.a_hp, signal)
        except Exception:
            return signal

    def _raise_alert(self, reason, source_name):
        now = time.monotonic()
        with self.alert_lock:
            if now - self.last_alert_time < self.cooldown:
                return
            self.last_alert_time = now

        message = f"{source_name}: {reason}"
        print("\n" + "=" * 55)
        print("[CANH BAO AM THANH]", message)
        print("=" * 55)

        if self.alert_callback:
            try:
                self.alert_callback(message, 0.0)
            except Exception as exc:
                print("[AUDIO CALLBACK ERROR]", exc)

    def _process_device(self, source_id, model):
        state = self.states[source_id]
        audio_queue = state["audio_queue"]
        source_name = state["name"]

        grammar_json = json.dumps(self.distress_phrases + ["[unk]"], ensure_ascii=False)
        recognizer = KaldiRecognizer(model, self.sample_rate, grammar_json)
        block_duration = self.blocksize / float(self.sample_rate)

        while not self.stop_event.is_set():
            try:
                audio = audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
                mono = mono.astype(np.float32)

                # 1. Lọc tiếng ù tần số thấp + khuếch đại tín hiệu
                filtered = self._apply_highpass_filter(mono)
                clipped = np.clip(filtered * self.gain, -1.0, 1.0)

                # 2. Phát hiện tiếng hét / âm thanh lớn bất thường (Adaptive RMS)
                rms = float(np.sqrt(np.mean(clipped ** 2)))
                if rms < self.min_scream_threshold:
                    state["noise_floor"] = 0.95 * state["noise_floor"] + 0.05 * rms

                dynamic_threshold = max(
                    self.min_scream_threshold,
                    state["noise_floor"] * self.scream_multiplier,
                )

                if rms >= dynamic_threshold:
                    state["loud_duration"] += block_duration
                else:
                    state["loud_duration"] = max(0.0, state["loud_duration"] - block_duration)

                if state["loud_duration"] >= self.scream_duration:
                    self._raise_alert(
                        "Phat hien tieng het hoac am thanh lon keo dai",
                        source_name,
                    )
                    state["loud_duration"] = 0.0

                # 3. Nhận diện từ khóa kêu cứu tiếng Việt qua Vosk
                pcm16 = (clipped * 32767).astype(np.int16)
                if recognizer.AcceptWaveform(pcm16.tobytes()):
                    recognized_text = json.loads(recognizer.Result()).get("text", "")
                else:
                    recognized_text = json.loads(recognizer.PartialResult()).get("partial", "")

                clean_text = recognized_text.lower().replace("[unk]", "").strip()
                if clean_text:
                    for phrase in self.distress_phrases:
                        if phrase in clean_text:
                            self._raise_alert(f"Phat hien loi keu cuu: '{phrase}'", source_name)
                            recognizer.Reset()
                            break

            except Exception as exc:
                print(f"[AUDIO PROCESS ERROR] {source_name}: {exc}")

    def _open_single_device(self, source_id, device, model):
        try:
            if device is not None:
                info = sd.query_devices(device, kind="input")
                dev_name = info.get("name", f"Mic {device}")
            else:
                dev_name = "Default Microphone"

            state = self._create_device_state(device, dev_name)
            self.states[source_id] = state

            stream = sd.InputStream(
                device=device,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.blocksize,
                callback=self._audio_callback(source_id),
            )
            stream.start()
            self.streams.append(stream)

            t = threading.Thread(
                target=self._process_device,
                args=(source_id, model),
                daemon=True,
            )
            t.start()
            self.processing_threads.append(t)
            print(f"[AUDIO] Da ket noi micro: {dev_name} (Index {device})")
            return True
        except Exception as exc:
            print(f"[AUDIO WARN] Bo qua micro index {device}: {exc}")
            return False

    def _run(self):
        try:
            model = Model(self.model_path)
            opened_count = 0

            for idx, device in enumerate(self.devices):
                if self.stop_event.is_set():
                    break
                if self._open_single_device(idx, device, model):
                    opened_count += 1

            # Nếu toàn bộ index chỉ định đều lỗi, tự động lùi về Micro mặc định của Windows
            if opened_count == 0 and self.devices != [None]:
                if self._open_single_device(99, None, model):
                    opened_count += 1

            if opened_count == 0:
                print("[AUDIO ERROR] Khong tim thay microphone nao kha dung.")
                return

            while not self.stop_event.wait(0.2):
                pass
        except Exception as exc:
            if not self.stop_event.is_set():
                print("[AUDIO ERROR]", exc)
        finally:
            self._close_streams()

    def _close_streams(self):
        for stream in self.streams:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self.streams.clear()

        for thread in self.processing_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        self.processing_threads.clear()

    def start(self):
        if self.worker and self.worker.is_alive():
            return

        self.stop_event.clear()
        self.states.clear()
        self.streams.clear()
        self.processing_threads.clear()

        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        for stream in self.streams:
            try:
                stream.stop()
            except Exception:
                pass

        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=4.0)

        self._close_streams()
        print("[AUDIO] Da dung giam sat am thanh.")