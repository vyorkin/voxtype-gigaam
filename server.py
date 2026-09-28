#!/usr/bin/env python3
"""OpenAI-compatible transcription server backed by local GigaAM / Parakeet.

Voxtype runs in `whisper.mode = "remote"` and POSTs recorded audio to
`/v1/audio/transcriptions` (multipart/form-data, field `file`). This server
answers with `{"text": "..."}`.

Two models are available:

* `gigaam-v3-e2e-rnnt` (int8) — Russian, punctuation and capitalization.
  Loaded at startup, it is the default for everything.
* `nemo-parakeet-tdt-0.6b-v3` (int8) — multilingual with good English.
  Loaded lazily on the first request that needs it.

The `model` form field selects what to use:

* a name containing `gigaam` — Russian model only;
* a name containing `parakeet`, or `en` — English model only;
* anything else (for example `auto`, the configured default) — the Russian
  model runs first and the English model is consulted when its output is not
  Cyrillic, which is what an English utterance looks like coming out of GigaAM.

Per-utterance latency is decode only (~0.1x realtime on CPU) because models are
loaded once and kept in memory.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.request
import wave
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnx_asr
import onnxruntime as ort

SAMPLE_RATE = 16000
DUMP_KEEP = 200
RUSSIAN_MODEL = "gigaam-v3-e2e-rnnt"
ENGLISH_MODEL = "nemo-parakeet-tdt-0.6b-v3"
QUANTIZATION = "int8"
# Share of Cyrillic letters in the output below which the utterance is treated
# as not-Russian and re-decoded with the English model.
CYRILLIC_THRESHOLD = 0.5
# Latin words in a Russian result mean code-switching ("... распознаётся
# English languigh."), which is where GigaAM mangles English. Such a phrase is
# re-decoded by whisper; see WhisperClient.
LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")
# whisper.cpp `whisper-server` that serves the multilingual second opinion. When
# empty, code-switched and English phrases fall back to the Parakeet model.
WHISPER_ENDPOINT = os.environ.get("VOXTYPE_WHISPER_ENDPOINT", "").strip()
WHISPER_TIMEOUT = float(os.environ.get("VOXTYPE_WHISPER_TIMEOUT", "30"))
WHISPER_MODEL_ID = os.environ.get("VOXTYPE_WHISPER_MODEL", "whisper-large-v3-turbo")


def cyrillic_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    cyrillic = sum(1 for char in letters if "\u0400" <= char <= "\u04ff")
    return cyrillic / len(letters)


def detect_language(text: str, fallback: str) -> str:
    """Language of the returned text, for the response and the dump metadata.

    The role that produced the text is kept only when the text has no letters
    (empty result), because the multilingual model happily returns Russian.
    """
    if not any(char.isalpha() for char in text):
        return fallback
    return "ru" if cyrillic_ratio(text) >= CYRILLIC_THRESHOLD else "en"


def route(model_field: str | None, english_enabled: bool) -> str:
    """Decide which model the request asks for: 'ru', 'en' or 'auto'."""
    name = (model_field or "").strip().lower()
    if "gigaam" in name:
        return "ru"
    if not english_enabled:
        return "ru"
    if "parakeet" in name or name in ("en", "english", "auto-en"):
        return "en"
    return "auto"


class Recognizer:
    """Thread-safe wrapper around one onnx-asr model, loaded on first use."""

    def __init__(self, model_id: str, threads: int, eager: bool) -> None:
        self.model_id = model_id
        self._threads = threads
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._model = None
        self.load_seconds = 0.0
        if eager:
            self.load()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            options = ort.SessionOptions()
            options.intra_op_num_threads = self._threads
            options.inter_op_num_threads = 1
            started = time.monotonic()
            # The timestamped adapter is used even though timestamps are not
            # needed: it is the one that returns per-token logprobs, which the
            # confidence check below relies on. It costs nothing extra.
            self._model = onnx_asr.load_model(
                self.model_id,
                quantization=QUANTIZATION,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            ).with_timestamps()
            self.load_seconds = time.monotonic() - started

    def warmup(self) -> None:
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> tuple[str, float]:
        """Return (text, mean token logprob). The logprob is NaN when absent."""
        self.load()
        with self._lock:
            assert self._model is not None
            result = self._model.recognize(samples, sample_rate=sample_rate)
        text = result.text.strip() if isinstance(result.text, str) else ""
        logprobs = getattr(result, "logprobs", None) or []
        confidence = float(np.mean(logprobs)) if logprobs else float("nan")
        return text, confidence


class WhisperClient:
    """Multilingual second opinion served by a local whisper.cpp server.

    GigaAM is the strongest local model for Russian, but it mangles English
    words inside Russian speech and cannot do English at all. whisper
    large-v3-turbo is the only local model that keeps embedded English in
    Latin script and spells it correctly, so every phrase that GigaAM returns
    without Cyrillic, or with Latin words in it, is re-decoded here.

    The model lives in a long-running `whisper-server` process (Vulkan), so a
    request pays only the decode. The `/inference` endpoint answers with
    `{"text": "..."}`.
    """

    def __init__(self, endpoint: str, timeout: float, model_id: str) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self.model_id = model_id

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        payload = io.BytesIO()
        with wave.open(payload, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(sample_rate)
            clipped = np.clip(samples, -1.0, 1.0)
            out.writeframes((clipped * 32767.0).astype("<i2").tobytes())

        boundary = "----voxtype" + os.urandom(12).hex()
        body = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n',
                b"Content-Type: audio/wav\r\n\r\n",
                payload.getvalue(),
                f"\r\n--{boundary}--\r\n".encode(),
            ]
        )
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            answer = json.loads(response.read())
        # whisper separates segments with newlines; voxtype turns a newline into
        # Enter (shift_enter_newlines is off), which sends the chat message
        # before the user asks. Dictation is a single line, so collapse it.
        return " ".join(str(answer.get("text", "")).split())


def audio_stats(samples: np.ndarray) -> tuple[float, float]:
    """Return (rms, peak) for logging and for the dump metadata."""
    if samples.size == 0:
        return 0.0, 0.0
    return float(np.sqrt((samples**2).mean())), float(np.abs(samples).max())


def dump_request(
    samples: np.ndarray,
    sample_rate: int,
    meta: dict,
    directory: str | None,
) -> None:
    """Save the received audio plus metadata when a dump directory is set.

    Used to debug accuracy problems on real speech: the exact audio that a
    transcription got wrong stays on disk, so alternative models and settings
    can be compared on it offline. Keeps the newest DUMP_KEEP pairs.
    """
    if not directory:
        return
    try:
        os.makedirs(directory, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = os.path.join(directory, f"{stamp}-{int(time.time() * 1000) % 1000:03d}")
        clipped = np.clip(samples, -1.0, 1.0)
        with wave.open(base + ".wav", "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(sample_rate)
            out.writeframes((clipped * 32767.0).astype("<i2").tobytes())
        with open(base + ".json", "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)

        existing = sorted(f for f in os.listdir(directory) if f.endswith(".wav"))
        for stale in existing[:-DUMP_KEEP]:
            for suffix in (".wav", ".json"):
                try:
                    os.remove(os.path.join(directory, stale[: -len(".wav")] + suffix))
                except OSError:
                    pass
    except Exception as exc:  # noqa: BLE001 - dumping must never break a request
        sys.stderr.write(f"dump failed: {type(exc).__name__}: {exc}\n")


def decode_wav(payload: bytes) -> tuple[np.ndarray, int]:
    """Decode a PCM WAV into float32 mono samples.

    Voxtype always sends 16 kHz mono PCM16, but the reader below accepts the
    wider set of PCM formats `wave` supports and averages extra channels, so a
    hand-run `curl` with a different WAV still works.
    """
    with wave.open(io.BytesIO(payload), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    if sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 1:
        data = (np.frombuffer(raw, dtype="<u1").astype(np.float32) - 128.0) / 128.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sample_width} bytes")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    return np.ascontiguousarray(data, dtype=np.float32), sample_rate


def parse_multipart(body: bytes, content_type: str) -> dict[str, tuple[str | None, bytes]]:
    """Extract form fields from a multipart/form-data body using only stdlib."""
    message = BytesParser(policy=email_policy).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )
    fields: dict[str, tuple[str | None, bytes]] = {}
    if not message.is_multipart():
        return fields
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        fields[name] = (part.get_filename(), payload)
    return fields


class Handler(BaseHTTPRequestHandler):
    server_version = "voxtype-gigaam"
    protocol_version = "HTTP/1.1"

    recognizers: dict[str, Recognizer]
    whisper: "WhisperClient | None"
    english_enabled: bool
    dump_dir: str | None
    started_at: float

    def _model_id(self, role: str) -> str:
        if role == "whisper":
            return self.whisper.model_id if self.whisper else WHISPER_MODEL_ID
        return self.recognizers[role].model_id

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/healthz"):
            self._send_json(
                200,
                {
                    "status": "ok",
                    "models": {
                        role: {
                            "id": rec.model_id,
                            "loaded": rec.loaded,
                            "load_seconds": round(rec.load_seconds, 3),
                        }
                        for role, rec in self.recognizers.items()
                    },
                    "english_enabled": self.english_enabled,
                    "whisper": self.whisper.endpoint if self.whisper else None,
                    "uptime_s": round(time.monotonic() - self.started_at, 1),
                },
            )
            return
        if path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": rec.model_id, "object": "model"}
                        for rec in self.recognizers.values()
                    ],
                },
            )
            return
        self._send_json(404, {"error": {"message": f"unknown path: {path}"}})

    def _transcribe_auto(
        self, samples: np.ndarray, sample_rate: int
    ) -> tuple[str, str, float]:
        """Russian first, with a multilingual second opinion for English.

        GigaAM handles pure Russian (with punctuation) and is the default.
        Everything else is re-decoded:

        1. No Cyrillic at all — the utterance is English. Parakeet is tried
           first: it is fast (CPU) and accurate for English, and never sees
           the code-switching case it is bad at.
        2. Latin words inside otherwise Russian text — code-switching, where
           GigaAM hears the English word but usually misspells it ("VoxTipe",
           "Rast и Tipecript"). whisper is the only local model that keeps
           the embedded English in Latin script and spells it correctly.

        A candidate wins only when it keeps at least as many Latin words as
        GigaAM, so a model that transliterates the English ("юзер ID" ->
        "юзер-ид") cannot replace a mostly-correct GigaAM result. Confidence
        is not used to choose between models: their mean token logprobs are
        not on the same scale (Parakeet scored ~0.09 higher on the same
        audio), so comparing them picked the wrong model most of the time.
        """
        text, confidence = self.recognizers["ru"].transcribe(samples, sample_rate)
        ratio = cyrillic_ratio(text)
        code_switched = bool(LATIN_WORD_RE.search(text))
        if not code_switched and ratio >= CYRILLIC_THRESHOLD:
            return text, "ru", confidence

        baseline = len(LATIN_WORD_RE.findall(text))

        def candidate(
            result: str, role: str, logprob: float
        ) -> tuple[str, str, float] | None:
            if result and len(LATIN_WORD_RE.findall(result)) >= baseline:
                self.log_message(
                    "second opinion via %s: ru=%r | %s=%r", role, text, role, result
                )
                return result, role, logprob
            return None

        if ratio < CYRILLIC_THRESHOLD and "en" in self.recognizers:
            english, english_confidence = self.recognizers["en"].transcribe(
                samples, sample_rate
            )
            accepted = candidate(english, "en", english_confidence)
            if accepted is not None:
                return accepted

        if self.whisper is not None:
            english = ""
            try:
                english = self.whisper.transcribe(samples, sample_rate)
            except Exception as exc:  # noqa: BLE001 - never fail the request
                self.log_message(
                    "whisper second opinion failed (%s: %s)", type(exc).__name__, exc
                )
            accepted = candidate(english, "whisper", confidence)
            if accepted is not None:
                return accepted

        return text, "ru", confidence

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path not in ("/v1/audio/transcriptions", "/v1/audio/translations"):
            self._send_json(404, {"error": {"message": f"unknown path: {path}"}})
            return

        content_type = self.headers.get("Content-Type") or ""
        try:
            body = self._read_body()
            requested = None
            if content_type.startswith("multipart/form-data"):
                fields = parse_multipart(body, content_type)
                if "file" not in fields:
                    self._send_json(
                        400, {"error": {"message": "multipart body has no 'file' field"}}
                    )
                    return
                audio = fields["file"][1]
                if "model" in fields:
                    requested = fields["model"][1].decode("utf-8", "replace")
            else:
                audio = body

            samples, sample_rate = decode_wav(audio)
            seconds = len(samples) / float(sample_rate)
            rms, peak = audio_stats(samples)
            if peak >= 0.999:
                hard = int((np.abs(samples) >= 0.999).sum())
                self.log_message(
                    "warning: input reaches full scale (%d samples at +/-1.0); "
                    "lower the microphone gain to avoid clipping",
                    hard,
                )
            target = route(requested, self.english_enabled)
            started = time.monotonic()
            if target == "ru":
                text, confidence = self.recognizers["ru"].transcribe(samples, sample_rate)
                used = "ru"
            elif target == "en":
                text, confidence = self.recognizers["en"].transcribe(samples, sample_rate)
                used = "en"
            else:
                text, used, confidence = self._transcribe_auto(samples, sample_rate)
            elapsed = time.monotonic() - started
            self.log_message(
                "transcribed %.2fs audio (rms=%.4f peak=%.3f) in %.2fs via %s "
                "(requested=%s, confidence=%+.3f, %d chars)",
                seconds,
                rms,
                peak,
                elapsed,
                used,
                requested or "default",
                confidence,
                len(text),
            )
            dump_request(
                samples,
                sample_rate,
                {
                    "text": text,
                    "model": self._model_id(used),
                    "language": detect_language(text, used),
                    "role": used,
                    "requested": requested or "default",
                    "seconds": round(seconds, 3),
                    "rms": round(rms, 5),
                    "peak": round(peak, 4),
                    "confidence": None if confidence != confidence else round(confidence, 4),
                    "decode_seconds": round(elapsed, 3),
                },
                self.dump_dir,
            )
            self._send_json(
                200,
                {
                    "text": text,
                    "model": self._model_id(used),
                    "language": detect_language(text, used),
                },
            )
        except Exception as exc:  # noqa: BLE001 - report every failure as JSON
            self.log_message("transcription failed: %s: %s", type(exc).__name__, exc)
            self._send_json(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})


def adopt_systemd_socket(server: ThreadingHTTPServer) -> socket.socket | None:
    """Use the listening socket passed by systemd socket activation, if any.

    With `voxtype-gigaam.socket` enabled the service is started on the first
    incoming connection instead of at login, so an idle machine pays no RAM
    for the models. Returns the adopted socket, or None to bind normally.
    """
    if int(os.environ.get("LISTEN_FDS", "0") or 0) < 1:
        return None
    inherited = socket.fromfd(3, socket.AF_INET, socket.SOCK_STREAM)
    inherited.setblocking(True)
    server.socket.close()
    server.socket = inherited
    print(f"adopted systemd socket fd=3: {inherited.getsockname()}", flush=True)
    return inherited


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("VOXTYPE_GIGAAM_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("VOXTYPE_GIGAAM_PORT", "9017"))
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("VOXTYPE_GIGAAM_THREADS", "6")),
        help="ONNX Runtime intra-op threads (default: 6)",
    )
    parser.add_argument(
        "--selftest",
        metavar="WAV",
        help="transcribe WAV with both models, print the result and exit",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help="with --selftest: auto (default), ru, en, or a Model name to send",
    )
    args = parser.parse_args()

    english_enabled = os.environ.get("VOXTYPE_GIGAAM_ENGLISH", "1") not in ("0", "false", "no")
    recognizers = {"ru": Recognizer(RUSSIAN_MODEL, args.threads, eager=True)}
    if english_enabled:
        recognizers["en"] = Recognizer(ENGLISH_MODEL, args.threads, eager=False)
    whisper = (
        WhisperClient(WHISPER_ENDPOINT, WHISPER_TIMEOUT, WHISPER_MODEL_ID)
        if WHISPER_ENDPOINT
        else None
    )

    if args.selftest:
        with open(args.selftest, "rb") as handle:
            samples, sample_rate = decode_wav(handle.read())
        target = route(args.model, english_enabled)
        roles = (["ru"] if target == "ru" else ["en"] if target == "en" else ["ru", "en"])
        for role in roles:
            started = time.monotonic()
            text, confidence = recognizers[role].transcribe(samples, sample_rate)
            print(f"[{role}] {time.monotonic() - started:.2f}s conf={confidence:+.3f} {text}")
        return 0

    recognizers["ru"].warmup()
    print(
        f"ready: ru={RUSSIAN_MODEL} (load {recognizers['ru'].load_seconds:.2f}s)"
        + (f" en={ENGLISH_MODEL} (lazy)" if english_enabled else " en=disabled")
        + (f" whisper={WHISPER_ENDPOINT}" if whisper else " whisper=disabled")
        + f" threads={args.threads}"
        + f" listen=http://{args.host}:{args.port}/v1/audio/transcriptions",
        flush=True,
    )

    Handler.recognizers = recognizers
    Handler.whisper = whisper
    Handler.english_enabled = english_enabled
    Handler.dump_dir = os.environ.get("VOXTYPE_GIGAAM_DUMP_DIR") or None
    if Handler.dump_dir:
        print(f"dumping received audio to {Handler.dump_dir}", flush=True)
    Handler.started_at = time.monotonic()
    server = ThreadingHTTPServer((args.host, args.port), Handler, bind_and_activate=False)
    server.daemon_threads = True
    if adopt_systemd_socket(server) is None:
        server.server_bind()
    server.server_activate()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
