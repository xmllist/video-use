"""Transcribe a video locally with WhisperX.

Extracts mono 16kHz audio via ffmpeg, runs WhisperX (Whisper ASR + forced
word-level alignment + optional pyannote speaker diarization), and writes a
**Scribe-compatible** JSON to <edit_dir>/transcripts/<video_stem>.json so the
rest of video-use (pack_transcripts.py, timeline_view.py, render.py) is
unchanged.

Output shape (matches the ElevenLabs Scribe schema the pipeline reads):
    {
      "language": "en",
      "text": "full transcript ...",
      "words": [
        {"type": "word",    "text": "Hello", "start": 0.00, "end": 0.32, "speaker_id": "speaker_0"},
        {"type": "spacing", "text": " ",     "start": 0.32, "end": 0.55, "speaker_id": null},
        ...
      ]
    }

Local, offline, no API key. Two tradeoffs vs hosted Scribe:
  - Speaker diarization (the `speaker_id` field) requires a HuggingFace token
    with access to `pyannote/speaker-diarization-3.1`. Set HF_TOKEN (or
    HUGGINGFACE_TOKEN / ELEVENLABS-style .env entry HF_TOKEN=...). Without a
    token, transcription still works; every `speaker_id` is null.
  - Whisper does not tag audio events ((laughs), (applause)) and tends to drop
    filler words (um/uh). The cut craft that leans on those signals degrades
    gracefully — silence gaps and word boundaries are still exact.

Cached: if the output file already exists, transcription is skipped.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit
    python helpers/transcribe.py <video_path> --language en
    python helpers/transcribe.py <video_path> --num-speakers 2
    python helpers/transcribe.py <video_path> --model large-v3 --no-diarize
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# Default Whisper model. large-v3 is the quality default; override with --model
# (e.g. "medium.en", "small") for faster CPU runs. Can also be set via the
# VIDEO_USE_WHISPER_MODEL env var.
DEFAULT_MODEL = os.environ.get("VIDEO_USE_WHISPER_MODEL", "large-v3")


def load_hf_token() -> str | None:
    """Token for pyannote diarization. Looked up in .env (repo root or cwd) then env.

    Recognized keys: HF_TOKEN, HUGGINGFACE_TOKEN, HUGGING_FACE_HUB_TOKEN.
    Returns None if not found — diarization is then skipped, not fatal.
    """
    keys = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    for candidate in [Path(__file__).resolve().parent.parent / ".env", Path(".env")]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() in keys:
                    val = v.strip().strip('"').strip("'")
                    if val:
                        return val
    for k in keys:
        v = os.environ.get(k, "")
        if v:
            return v
    return None


def pick_device() -> tuple[str, str]:
    """Return (device, compute_type) for the ASR model.

    WhisperX's ASR backend is faster-whisper (CTranslate2), which supports CUDA
    and CPU but not Apple MPS — so on Apple Silicon we run the ASR on CPU. The
    alignment + diarization torch models can still use MPS, but to keep things
    robust and predictable we keep everything on the same device.
    """
    try:
        import torch
    except Exception:
        return "cpu", "int8"
    if torch.cuda.is_available():
        return "cuda", "float16"
    # Apple Silicon / CPU: CTranslate2 has no MPS path. int8 keeps CPU usable.
    return "cpu", "int8"


def extract_audio(video_path: Path, dest: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _norm_speaker(label: str | None) -> str | None:
    """Map WhisperX speaker labels (SPEAKER_00) to Scribe-style (speaker_0).

    pack_transcripts.py strips the "speaker_" prefix and renders "S0".
    """
    if not label:
        return None
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    if digits == "":
        return str(label)
    return f"speaker_{int(digits)}"


def whisperx_to_scribe(result: dict) -> dict:
    """Convert a WhisperX aligned (+optionally diarized) result to Scribe schema.

    WhisperX gives `segments`, each with a `words` list of
    {word, start, end, score, speaker?}. We flatten to a single `words` array,
    inserting `spacing` entries for the gaps between consecutive words so the
    downstream silence logic has explicit gap markers (it also independently
    derives gaps from word boundaries, so this is belt-and-suspenders).
    """
    words_out: list[dict] = []
    text_parts: list[str] = []
    prev_end: float | None = None

    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = (w.get("word") or "").strip()
            if not text:
                continue
            start = w.get("start")
            end = w.get("end")
            speaker = _norm_speaker(w.get("speaker"))

            # Explicit spacing token for any real gap before this word.
            if (
                start is not None
                and prev_end is not None
                and start - prev_end > 1e-3
            ):
                words_out.append({
                    "type": "spacing",
                    "text": " ",
                    "start": round(prev_end, 3),
                    "end": round(start, 3),
                    "speaker_id": None,
                })

            words_out.append({
                "type": "word",
                "text": text,
                "start": round(start, 3) if start is not None else None,
                "end": round(end, 3) if end is not None else None,
                "speaker_id": speaker,
            })
            text_parts.append(text)
            if end is not None:
                prev_end = end

    return {
        "language": result.get("language"),
        "text": " ".join(text_parts),
        "words": words_out,
    }


def load_model(model_size: str = DEFAULT_MODEL):
    """Load and return a WhisperX ASR model (reuse across files in batch mode)."""
    import whisperx

    device, compute_type = pick_device()
    return whisperx.load_model(model_size, device, compute_type=compute_type)


def transcribe_one(
    video: Path,
    edit_dir: Path,
    model=None,
    language: str | None = None,
    num_speakers: int | None = None,
    diarize: bool = True,
    hf_token: str | None = None,
    verbose: bool = True,
) -> Path:
    """Transcribe a single video with WhisperX. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    `model` may be a preloaded WhisperX model (batch mode) or None to load one.
    """
    import whisperx

    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video.stem}.json"

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    device, _ = pick_device()
    if model is None:
        if verbose:
            print(f"  loading WhisperX model ({DEFAULT_MODEL}) on {device}", flush=True)
        model = load_model()

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / f"{video.stem}.wav"
        if verbose:
            print(f"  extracting audio from {video.name}", flush=True)
        extract_audio(video, audio_path)

        audio = whisperx.load_audio(str(audio_path))

        if verbose:
            print(f"  transcribing {video.stem}", flush=True)
        result = model.transcribe(audio, language=language, batch_size=16)
        detected_lang = result.get("language", language)

        # Forced alignment → word-level timestamps.
        if verbose:
            print(f"  aligning words ({detected_lang})", flush=True)
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_lang, device=device
            )
            result = whisperx.align(
                result["segments"], align_model, metadata, audio, device,
                return_char_alignments=False,
            )
            result["language"] = detected_lang
        except Exception as e:
            if verbose:
                print(f"  alignment unavailable for '{detected_lang}' ({e}); "
                      f"using segment-level times", flush=True)
            # Fall back to segment timings as word-ish entries.
            for seg in result.get("segments", []):
                seg.setdefault("words", [{
                    "word": seg.get("text", "").strip(),
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                }])

        # Optional speaker diarization.
        if diarize:
            token = hf_token or load_hf_token()
            if not token:
                if verbose:
                    print("  diarization skipped: no HuggingFace token "
                          "(set HF_TOKEN in .env to enable speaker labels)", flush=True)
            else:
                try:
                    if verbose:
                        print("  diarizing speakers", flush=True)
                    # In whisperx 3.8.x the pipeline lives under whisperx.diarize
                    # and takes `token=` (not `use_auth_token=`). It feeds an
                    # in-memory waveform to pyannote, so torchcodec is bypassed.
                    from whisperx.diarize import DiarizationPipeline
                    dia = DiarizationPipeline(token=token, device=device)
                    dia_kwargs = {}
                    if num_speakers:
                        dia_kwargs["num_speakers"] = num_speakers
                    diarize_segments = dia(audio, **dia_kwargs)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                except Exception as e:
                    if verbose:
                        print(f"  diarization failed ({e}); continuing without "
                              f"speaker labels", flush=True)

    payload = whisperx_to_scribe(result)
    out_path.write_text(json.dumps(payload, indent=2))
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        n_words = sum(1 for w in payload["words"] if w.get("type") == "word")
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        print(f"    words: {n_words}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video locally with WhisperX")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g., 'en'). Omit to auto-detect.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers when known. Improves diarization accuracy.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Whisper model size (default: {DEFAULT_MODEL}). "
             f"Use medium.en / small for faster CPU runs.",
    )
    ap.add_argument(
        "--no-diarize",
        action="store_true",
        help="Skip speaker diarization even if a HuggingFace token is available.",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    model = load_model(args.model)
    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        model=model,
        language=args.language,
        num_speakers=args.num_speakers,
        diarize=not args.no_diarize,
    )


if __name__ == "__main__":
    main()
