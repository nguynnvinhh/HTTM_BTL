
import json
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer


class AudioDistressMonitor:
    def __init__(
        self,
        model_path="models/vosk-vn",
        alert_callback=None,
        sample_rate=16000,
        cooldown=8,
        scream_threshold=0.18,
        scream_duration=0.8,
        device=None,
    ):
        self.model_path = model_path
        self.alert_callback = alert_callback
        self.sample_rate = sample_rate
        self.cooldown = cooldown
        self.scream_threshold = scream_threshold
        self.scream_duration = scream_duration
        self.device = device

        self.audio_queue = queue.Queue(maxsize=40)
        self.stop_event = threading.Event()
        self.last_alert_time = 0
        self.loud_duration = 0.0
        self.worker = None

        self.distress_phrases = [
            "cứu tôi",
            "cứu với",
            "cứu",
            "giúp tôi",
            "giúp với",
            "help me",
            "help",
            "đau quá",
        ]

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
        recognizer = KaldiRecognizer(model, self.sample_rate)

        block_duration = 4000 / self.sample_rate

        print("[AUDIO] Micro dang hoat dong.")

        while not self.stop_event.is_set():
            try:
                audio = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            mono = audio[:, 0]
            rms = float(np.sqrt(np.mean(mono ** 2)))

            if rms >= self.scream_threshold:
                self.loud_duration += block_duration
            else:
                self.loud_duration = max(
                    0.0,
                    self.loud_duration - block_duration
                )

            if self.loud_duration >= self.scream_duration:
                self._raise_alert(
                    "Phat hien tieng het hoac am thanh lon keo dai"
                )
                self.loud_duration = 0.0

            pcm16 = (
                np.clip(mono, -1.0, 1.0) * 32767
            ).astype(np.int16)

            if recognizer.AcceptWaveform(pcm16.tobytes()):
                result = json.loads(recognizer.Result())
                recognized = result.get("text", "").lower().strip()

                if not recognized:
                    continue

                print("[AUDIO] Nhan dien:", recognized)

                if any(
                    phrase in recognized
                    for phrase in self.distress_phrases
                ):
                    self._raise_alert(
                        f"Phat hien loi keu cuu: {recognized}"
                    )

    def _run(self):
        try:
            # Nạp mô hình trong luồng riêng để không chặn camera
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

            # Thu mo device duoc chi dinh (hoac device 3 neu de mac dinh), fallback ve None (thiet bi mac dinh cua OS)
            stream = None
            for dev in (self.device, 3, None):
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
