"""Free, offline transcription backends that write Scribe-shaped JSON.

Every backend returns the same dict shape ElevenLabs Scribe does — a `words`
list of {text, start, end, type, speaker_id} with type 'word' or
'audio_event' — so pack_transcripts.py, timeline_view.py and render.py read
local transcripts exactly like hosted ones.

Backends:
    sensevoice  SenseVoice-Small (zh/en/ja/ko/yue) via sherpa-onnx, with
                Silero VAD. CPU, ~20-30x realtime. Model weights come from
                GitHub releases, so it works in sandboxes that block
                Hugging Face (e.g. Claude Code on the web).
    whisper     faster-whisper with native word timestamps. Better accuracy
                and ~100 languages, but the weights live on Hugging Face —
                needs huggingface.co reachable (or a pre-filled HF cache).
    activity    No speech recognition at all. Labels where there is speech,
                other sound, and silence. Needs only numpy; uses Silero VAD
                when it can get it, plain loudness otherwise.

All backends also tag loud non-speech stretches between utterances as audio
events ("(noise -28dB)", "(tonal sound -31dB)"), so the editor can tell a
quiet gap from a gap full of room noise or music.

Models download once to $VIDEO_USE_MODELS (default ~/.cache/video-use/models).

Usage (normally through transcribe.py --backend ...):
    python helpers/transcribe_local.py <audio.wav> --backend sensevoice
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16000

GITHUB_ASR = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
SENSEVOICE_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
SENSEVOICE_URL = f"{GITHUB_ASR}/{SENSEVOICE_NAME}.tar.bz2"
VAD_URL = f"{GITHUB_ASR}/silero_vad.onnx"

SENSEVOICE_LANGS = {"auto", "zh", "en", "ja", "ko", "yue"}
# Scripts written without spaces: every SenseVoice token is its own word.
NO_SPACE_LANGS = {"zh", "ja", "yue"}

# SenseVoice event tags -> Scribe-style audio event text.
EVENT_LABELS = {
    "BGM": "music",
    "Applause": "applause",
    "Laughter": "laughter",
    "Cry": "crying",
    "Sneeze": "sneeze",
    "Breath": "breath",
    "Cough": "cough",
}

PUNCT = set(",.?!;:、，。？！；：…\"'")

# A CTC token timestamp marks where a token starts, not where it ends. A word
# ends at the next word's start, but never runs longer than this — otherwise a
# word before a mid-utterance pause would swallow the pause.
MAX_WORD_S = 0.7

# Non-speech sound detection.
FRAME_S = 0.05
SOUND_FLOOR_DB = -45.0        # quieter than this counts as silence
SOUND_MIN_S = 0.3             # shorter non-speech blips are ignored
SOUND_MERGE_GAP_S = 0.25
NOISE_FLATNESS = 0.3          # spectral flatness above this reads as noise


# -------- model files --------------------------------------------------------


def models_dir() -> Path:
    d = Path(os.environ.get("VIDEO_USE_MODELS", Path.home() / ".cache" / "video-use" / "models"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, dest: Path) -> None:
    import requests

    tmp = dest.with_name(dest.name + ".part")
    print(f"  downloading {url}", file=sys.stderr, flush=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)


def ensure_vad() -> Path:
    path = models_dir() / "silero_vad.onnx"
    if not path.exists():
        _download(VAD_URL, path)
    return path


def ensure_sensevoice() -> Path:
    root = models_dir()
    model_dir = root / SENSEVOICE_NAME
    if (model_dir / "model.int8.onnx").exists() and (model_dir / "tokens.txt").exists():
        return model_dir
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        archive = Path(tmp) / "model.tar.bz2"
        _download(SENSEVOICE_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(tmp)
        shutil.rmtree(model_dir, ignore_errors=True)
        (Path(tmp) / SENSEVOICE_NAME).rename(model_dir)
    return model_dir


# -------- audio --------------------------------------------------------------


def load_wav(path: Path) -> np.ndarray:
    """16 kHz mono 16-bit wav (what transcribe.extract_audio writes) -> float32."""
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path.name}: expected 16 kHz mono 16-bit PCM")
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _frame_stats(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame loudness (dBFS) and spectral flatness."""
    hop = int(FRAME_S * SAMPLE_RATE)
    n = len(samples) // hop
    if n == 0:
        return np.zeros(0), np.zeros(0)
    frames = samples[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    spec = np.abs(np.fft.rfft(frames * np.hanning(hop), axis=1)) + 1e-10
    flatness = np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1)
    return db, flatness


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _merge(spans: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in sorted(spans):
        if out and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


# -------- speech segmentation -------------------------------------------------


def vad_segments(samples: np.ndarray, max_speech_s: float = 20.0) -> list[tuple[float, float]]:
    """Speech spans in seconds via Silero VAD (sherpa-onnx)."""
    import sherpa_onnx

    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(ensure_vad())
    cfg.silero_vad.threshold = 0.5
    cfg.silero_vad.min_silence_duration = 0.3
    cfg.silero_vad.min_speech_duration = 0.2
    cfg.silero_vad.max_speech_duration = max_speech_s
    cfg.sample_rate = SAMPLE_RATE
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=max_speech_s + 10)

    spans: list[tuple[float, float]] = []

    def drain() -> None:
        while not vad.empty():
            seg = vad.front
            start = seg.start / SAMPLE_RATE
            spans.append((start, start + len(seg.samples) / SAMPLE_RATE))
            vad.pop()

    window = cfg.silero_vad.window_size
    for i in range(0, len(samples), window):
        vad.accept_waveform(samples[i : i + window])
        drain()
    vad.flush()
    drain()
    return spans


def energy_segments(samples: np.ndarray) -> list[tuple[float, float]]:
    """Fallback 'activity' spans from loudness alone — no model needed."""
    db, _ = _frame_stats(samples)
    spans = [(s * FRAME_S, e * FRAME_S) for s, e in _runs(db > SOUND_FLOOR_DB)]
    return [(s, e) for s, e in _merge(spans, SOUND_MERGE_GAP_S) if e - s >= SOUND_MIN_S]


def sound_events(
    samples: np.ndarray,
    speech: list[tuple[float, float]],
    speaker: str = "speaker_0",
) -> list[dict]:
    """Audio events for audible stretches that are not speech."""
    db, flat = _frame_stats(samples)
    if not len(db):
        return []
    times = np.arange(len(db)) * FRAME_S
    in_speech = np.zeros(len(db), dtype=bool)
    for s, e in speech:
        in_speech |= (times + FRAME_S > s) & (times < e)
    loud = (db > SOUND_FLOOR_DB) & ~in_speech

    events: list[dict] = []
    spans = _merge([(s * FRAME_S, e * FRAME_S) for s, e in _runs(loud)], SOUND_MERGE_GAP_S)
    for s, e in spans:
        if e - s < SOUND_MIN_S:
            continue
        sel = (times >= s) & (times < e) & ~in_speech
        if not sel.any():
            continue
        level = float(np.percentile(db[sel], 90))
        kind = "noise" if float(np.median(flat[sel])) > NOISE_FLATNESS else "tonal sound"
        events.append({
            "text": f"({kind} {level:.0f}dB)",
            "start": round(s, 3),
            "end": round(e, 3),
            "type": "audio_event",
            "speaker_id": speaker,
        })
    return events


# -------- backends -------------------------------------------------------------


_lock = threading.Lock()
_recognizers: dict[tuple, object] = {}


def _sensevoice_recognizer(language: str):
    import sherpa_onnx

    key = ("sensevoice", language)
    with _lock:
        if key not in _recognizers:
            d = ensure_sensevoice()
            _recognizers[key] = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(d / "model.int8.onnx"),
                tokens=str(d / "tokens.txt"),
                num_threads=os.cpu_count() or 4,
                language=language,
                use_itn=True,
            )
        return _recognizers[key]


def _tokens_to_words(text: str, tokens: list[str], stamps: list[float], offset: float,
                     seg_end: float, no_space: bool, speaker: str) -> list[dict]:
    """Word entries from a SenseVoice result.

    Token spacing is unreliable for Korean ('조', ' 금', '만' for "조금만"),
    but the sentence text is not. When the text spells the same characters
    as the tokens, take word boundaries from the text and times from the
    tokens; otherwise fall back to the token spacing.
    """
    char_times = [offset + float(ts) for tok, ts in zip(tokens, stamps) for c in tok if not c.isspace()]
    pieces = text.split()
    if not no_space and "".join(pieces) == "".join(t for t in tokens if not t.isspace()).replace(" ", ""):
        # Trailing punctuation is emitted late; its timestamp is not speech,
        # so a word's last time is that of its last non-punctuation char.
        spans, i = [], 0
        for p in pieces:
            core = len(p.rstrip("".join(PUNCT))) or len(p)
            spans.append((p, char_times[i], char_times[i + core - 1]))
            i += len(p)
    else:
        spans, new_word = [], True
        for tok, ts in zip(tokens, stamps):
            piece = tok.strip()
            if not piece:
                new_word = True
                continue
            t = offset + float(ts)
            if spans and all(c in PUNCT for c in piece):
                p, s, last = spans[-1]
                spans[-1] = (p + piece, s, last)
            elif no_space or new_word or tok.startswith(" ") or not spans:
                spans.append((piece, t, t))
            else:
                p, s, _ = spans[-1]
                spans[-1] = (p + piece, s, t)
            new_word = False

    words: list[dict] = []
    for p, s, last in spans:
        if all(c in PUNCT for c in p):
            if words:
                words[-1]["text"] += p
            continue
        words.append({"text": p, "start": s, "_last": last, "type": "word", "speaker_id": speaker})
    for i, w in enumerate(words):
        nxt = words[i + 1]["start"] if i + 1 < len(words) else seg_end
        # A word lasts at least through its last token (+ one CTC frame).
        floor = w.pop("_last", w["start"]) + 0.12
        w["end"] = round(min(nxt, max(floor, min(w["start"] + MAX_WORD_S, nxt))), 3)
        w["start"] = round(w["start"], 3)
        if w["end"] <= w["start"]:
            w["end"] = round(w["start"] + 0.06, 3)
    return words


def transcribe_sensevoice(samples: np.ndarray, language: str | None = None) -> dict:
    lang = (language or "auto").lower()
    if lang not in SENSEVOICE_LANGS:
        raise ValueError(f"SenseVoice supports {sorted(SENSEVOICE_LANGS)}, not {lang!r}. "
                         "Use --backend whisper for other languages.")
    rec = _sensevoice_recognizer(lang)
    spans = vad_segments(samples)

    words: list[dict] = []
    langs: dict[str, float] = {}
    texts: list[str] = []
    for start, end in spans:
        chunk = samples[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
        stream = rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, chunk)
        rec.decode_stream(stream)
        res = stream.result
        seg_lang = res.lang.strip("<|>") or lang
        langs[seg_lang] = langs.get(seg_lang, 0.0) + (end - start)

        event = res.event.strip("<|>")
        if event in EVENT_LABELS:
            words.append({"text": f"({EVENT_LABELS[event]})", "start": round(start, 3),
                          "end": round(end, 3), "type": "audio_event", "speaker_id": "speaker_0"})
        if not res.text.strip():
            continue
        texts.append(res.text.strip())
        words.extend(_tokens_to_words(res.text, list(res.tokens), list(res.timestamps), start, end,
                                      seg_lang in NO_SPACE_LANGS, "speaker_0"))

    words.extend(sound_events(samples, spans))
    words.sort(key=lambda w: (w["start"], w["type"] != "audio_event"))
    return {
        "language_code": max(langs, key=langs.get) if langs else lang,
        "text": " ".join(texts),
        "words": words,
        "backend": "sensevoice",
        "model": SENSEVOICE_NAME,
    }


def transcribe_whisper(samples: np.ndarray, language: str | None = None,
                       model_size: str | None = None) -> dict:
    from faster_whisper import WhisperModel

    size = model_size or os.environ.get("VIDEO_USE_WHISPER_MODEL", "small")
    key = ("whisper", size)
    with _lock:
        if key not in _recognizers:
            try:
                _recognizers[key] = WhisperModel(
                    size, device="cpu", compute_type="int8",
                    download_root=str(models_dir() / "whisper"),
                )
            except Exception as e:
                raise RuntimeError(
                    f"could not load Whisper '{size}' ({e}). The weights come from "
                    "huggingface.co — allow that host in the network settings, or use "
                    "--backend sensevoice."
                ) from e
        model = _recognizers[key]

    segments, info = model.transcribe(
        samples, language=language, word_timestamps=True, vad_filter=True,
        # Keep 'um', 'uh' etc. instead of normalizing them away.
        initial_prompt="Umm, uh, so, like, you know, I mean... 음, 어, 그러니까, 아.",
    )
    words: list[dict] = []
    texts: list[str] = []
    speech: list[tuple[float, float]] = []
    for seg in segments:
        texts.append(seg.text.strip())
        speech.append((seg.start, seg.end))
        for w in seg.words or []:
            if not w.word.strip():
                continue
            words.append({"text": w.word.strip(), "start": round(w.start, 3),
                          "end": round(w.end, 3), "type": "word", "speaker_id": "speaker_0"})
    words.extend(sound_events(samples, speech))
    words.sort(key=lambda w: (w["start"], w["type"] != "audio_event"))
    return {
        "language_code": info.language,
        "text": " ".join(texts),
        "words": words,
        "backend": "whisper",
        "model": size,
    }


def transcribe_activity(samples: np.ndarray, language: str | None = None) -> dict:
    """Where is the speech, where is other sound, where is silence. No words."""
    try:
        speech = vad_segments(samples)
        how = "silero-vad"
    except Exception as e:  # no sherpa-onnx or no network for the VAD model
        print(f"  VAD unavailable ({e}); falling back to loudness only", file=sys.stderr)
        speech, how = energy_segments(samples), "loudness"

    db, _ = _frame_stats(samples)
    words: list[dict] = []
    for s, e in speech:
        i, j = int(s / FRAME_S), max(int(e / FRAME_S), int(s / FRAME_S) + 1)
        level = float(np.percentile(db[i:j], 90)) if j <= len(db) and i < j else 0.0
        label = "speech" if how == "silero-vad" else "sound"
        words.append({"text": f"({label} {level:.0f}dB)", "start": round(s, 3),
                      "end": round(e, 3), "type": "audio_event", "speaker_id": "speaker_0"})
    if how == "silero-vad":
        words.extend(sound_events(samples, speech))
    words.sort(key=lambda w: w["start"])
    return {
        "language_code": language or "",
        "text": "",
        "words": words,
        "backend": "activity",
        "model": how,
    }


BACKENDS = {
    "sensevoice": transcribe_sensevoice,
    "whisper": transcribe_whisper,
    "activity": transcribe_activity,
}


def resolve_backend(name: str) -> str:
    """'auto' -> the best local backend that is installed."""
    if name != "auto":
        return name
    try:
        import sherpa_onnx  # noqa: F401
        return "sensevoice"
    except ImportError:
        pass
    try:
        import faster_whisper  # noqa: F401
        return "whisper"
    except ImportError:
        return "activity"


def transcribe_wav(wav: Path, backend: str, language: str | None = None) -> dict:
    return BACKENDS[resolve_backend(backend)](load_wav(wav), language)


def main() -> None:
    ap = argparse.ArgumentParser(description="Local transcription of a 16 kHz mono wav")
    ap.add_argument("wav", type=Path, nargs="?")
    ap.add_argument("--backend", default="auto", choices=["auto", *BACKENDS])
    ap.add_argument("--language", default=None)
    ap.add_argument("--download-only", action="store_true",
                    help="Fetch the backend's model files and exit (for setup scripts).")
    args = ap.parse_args()
    if args.download_only:
        ensure_vad()
        if resolve_backend(args.backend) == "sensevoice":
            ensure_sensevoice()
        print(f"models ready in {models_dir()}")
        return
    if args.wav is None:
        ap.error("wav is required unless --download-only")
    print(json.dumps(transcribe_wav(args.wav, args.backend, args.language), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
