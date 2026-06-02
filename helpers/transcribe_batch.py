"""Batch-transcribe every video in a directory with local WhisperX.

Walks <videos_dir> for common video extensions, transcribes each with WhisperX
(word-level timestamps + optional diarization), and writes Scribe-compatible
transcripts to <videos_dir>/edit/transcripts/<name>.json.

The Whisper model is loaded ONCE and reused across all files — model load is
the expensive part, and CPU ASR is the bottleneck, so files are processed
sequentially with a shared model rather than spawning N model copies.

Cached per-file: any source that already has a transcript is skipped.

Usage:
    python helpers/transcribe_batch.py <videos_dir>
    python helpers/transcribe_batch.py <videos_dir> --num-speakers 2
    python helpers/transcribe_batch.py <videos_dir> --model medium.en
    python helpers/transcribe_batch.py <videos_dir> --no-diarize
    python helpers/transcribe_batch.py <videos_dir> --edit-dir /custom/edit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from transcribe import DEFAULT_MODEL, load_hf_token, load_model, transcribe_one


VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV", ".avi", ".AVI", ".m4v"}


def find_videos(videos_dir: Path) -> list[Path]:
    videos = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix in VIDEO_EXTS
    )
    return videos


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch transcription of a videos directory with WhisperX")
    ap.add_argument("videos_dir", type=Path, help="Directory containing source videos")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <videos_dir>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code. Omit to auto-detect per file.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers. Improves diarization when known.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Whisper model size (default: {DEFAULT_MODEL}).",
    )
    ap.add_argument(
        "--no-diarize",
        action="store_true",
        help="Skip speaker diarization even if a HuggingFace token is available.",
    )
    args = ap.parse_args()

    videos_dir = args.videos_dir.resolve()
    if not videos_dir.is_dir():
        sys.exit(f"not a directory: {videos_dir}")

    edit_dir = (args.edit_dir or (videos_dir / "edit")).resolve()
    (edit_dir / "transcripts").mkdir(parents=True, exist_ok=True)

    videos = find_videos(videos_dir)
    if not videos:
        sys.exit(f"no videos found in {videos_dir}")

    already_cached = [v for v in videos if (edit_dir / "transcripts" / f"{v.stem}.json").exists()]
    pending = [v for v in videos if v not in already_cached]

    print(f"found {len(videos)} videos ({len(already_cached)} cached, {len(pending)} to transcribe)")
    if not pending:
        print("nothing to do")
        return

    diarize = not args.no_diarize
    if diarize and not load_hf_token():
        print("note: no HuggingFace token found — speaker labels disabled "
              "(set HF_TOKEN in .env to enable diarization)")

    print(f"loading WhisperX model ({args.model}) once for {len(pending)} files")
    model = load_model(args.model)

    t0 = time.time()
    errors: list[tuple[Path, str]] = []
    for v in pending:
        try:
            out = transcribe_one(
                video=v,
                edit_dir=edit_dir,
                model=model,
                language=args.language,
                num_speakers=args.num_speakers,
                diarize=diarize,
                verbose=True,
            )
            print(f"  + {v.stem}  →  {out.name}")
        except Exception as e:
            errors.append((v, str(e)))
            print(f"  x {v.stem}  FAILED: {e}")

    dt = time.time() - t0
    print(f"\ndone in {dt:.1f}s")
    if errors:
        print(f"{len(errors)} failures:")
        for v, msg in errors:
            print(f"  {v.name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
