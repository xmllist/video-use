"""Generate a TTS voiceover with OmniVoice and (optionally) lay it onto a video.

OmniVoice (https://github.com/k2-fsa/OmniVoice) is a zero-shot multilingual
text-to-speech model. This helper is the bridge between it and a video edit:
write a script, get narration audio, and optionally mux it onto a clip — handy
for narrating silent process/timelapse videos or adding a voiceover to a montage.

OmniVoice is an OPTIONAL external engine (like the animation engines). It is not
a dependency of video-use; it runs in its own environment. Point this helper at
your OmniVoice checkout with the OMNIVOICE_HOME env var (the dir containing its
`.venv/`), or put `omnivoice-infer` on PATH, or set OMNIVOICE_BIN.

Three voice modes (mutually exclusive; default = auto voice):
  - auto:          just --text (model picks a voice)
  - voice design:  --instruct "female, british accent"
  - voice cloning: --ref-audio ref.wav [--ref-text "transcript of ref"]

Usage:
    # Generate narration WAV only
    python helpers/voiceover.py --text "Watch this teapot come to life." \
        --edit-dir /path/to/videos/edit -o narration.wav

    # From a script file, voice-designed, fit to a target duration
    python helpers/voiceover.py --text-file script.txt --instruct "warm female, slow" \
        --duration 105 -o narration.wav

    # Generate AND lay onto a video (duck original audio under the VO)
    python helpers/voiceover.py --text "..." --onto edit/final.mp4 \
        -o edit/final_vo.mp4 --mix --duck 0.15

    # Print the OmniVoice command without running it
    python helpers/voiceover.py --text "hi" -o out.wav --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_omnivoice_infer() -> str:
    """Locate the `omnivoice-infer` entry point. Raises with guidance if absent.

    Resolution order:
      1. OMNIVOICE_BIN (explicit path to the console script)
      2. OMNIVOICE_HOME/.venv/bin/omnivoice-infer
      3. ~/Developer/OmniVoice, ~/OmniVoice (.venv/bin/omnivoice-infer)
      4. omnivoice-infer on PATH
    """
    explicit = os.environ.get("OMNIVOICE_BIN")
    if explicit and Path(explicit).exists():
        return explicit

    candidates: list[Path] = []
    home = os.environ.get("OMNIVOICE_HOME")
    if home:
        candidates.append(Path(home) / ".venv" / "bin" / "omnivoice-infer")
    for base in (Path.home() / "Developer" / "OmniVoice", Path.home() / "OmniVoice"):
        candidates.append(base / ".venv" / "bin" / "omnivoice-infer")
    for c in candidates:
        if c.exists():
            return str(c)

    on_path = shutil.which("omnivoice-infer")
    if on_path:
        return on_path

    sys.exit(
        "OmniVoice not found. Install it (https://github.com/k2-fsa/OmniVoice) "
        "and either:\n"
        "  - export OMNIVOICE_HOME=/path/to/OmniVoice   (dir containing .venv/), or\n"
        "  - export OMNIVOICE_BIN=/path/to/omnivoice-infer, or\n"
        "  - put `omnivoice-infer` on PATH."
    )


def read_text(args) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    return args.text or ""


def generate_narration(
    infer_bin: str,
    text: str,
    out_wav: Path,
    *,
    model: str = "k2-fsa/OmniVoice",
    instruct: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    language: str | None = None,
    duration: float | None = None,
    speed: float | None = None,
    device: str | None = None,
    dry_run: bool = False,
) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [infer_bin, "--model", model, "--text", text, "--output", str(out_wav)]
    if instruct:
        cmd += ["--instruct", instruct]
    if ref_audio:
        cmd += ["--ref_audio", str(ref_audio)]
        if ref_text:
            cmd += ["--ref_text", ref_text]
    if language:
        cmd += ["--language", language]
    if duration is not None:
        cmd += ["--duration", f"{duration}"]
    if speed is not None:
        cmd += ["--speed", f"{speed}"]
    if device:
        cmd += ["--device", device]

    printable = " ".join(
        (f'"{c}"' if " " in c else c) for c in cmd
    )
    print(f"  $ {printable}")
    if dry_run:
        print("  (dry-run: not executed)")
        return out_wav

    subprocess.run(cmd, check=True)
    if not out_wav.exists():
        sys.exit(f"OmniVoice did not produce {out_wav}")
    print(f"narration → {out_wav}")
    return out_wav


def mux_onto_video(
    video: Path,
    narration_wav: Path,
    out_path: Path,
    *,
    mix: bool = False,
    duck: float = 0.15,
    vo_gain: float = 1.0,
    offset: float = 0.0,
) -> None:
    """Lay narration onto a video.

    mix=False (default): replace the audio track with the narration.
    mix=True: keep the original audio ducked to `duck` (0..1) under the
              narration at `vo_gain`. `offset` delays the VO start (seconds).
    Narration is resampled to 48 kHz stereo. Video is copied (no re-encode).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    delay_ms = int(max(0.0, offset) * 1000)
    vo_chain = (
        f"[1:a]aresample=48000,aformat=channel_layouts=stereo,"
        f"volume={vo_gain}"
        + (f",adelay={delay_ms}|{delay_ms}" if delay_ms else "")
        + "[vo]"
    )

    if mix:
        filter_complex = (
            f"{vo_chain};"
            f"[0:a]aresample=48000,aformat=channel_layouts=stereo,volume={duck}[bg];"
            f"[bg][vo]amix=inputs=2:duration=first:dropout_transition=0,"
            f"aresample=48000[aout]"
        )
    else:
        # Replace original audio with the VO, but keep the video's full length
        # by mixing VO over silence derived from the original track.
        filter_complex = (
            f"{vo_chain};"
            f"[0:a]aresample=48000,aformat=channel_layouts=stereo,volume=0[bg];"
            f"[bg][vo]amix=inputs=2:duration=first:dropout_transition=0,"
            f"aresample=48000[aout]"
        )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video),
        "-i", str(narration_wav),
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"muxing voiceover → {out_path.name} (mix={'on' if mix else 'off'})")
    subprocess.run(cmd, check=True)
    print(f"done: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate an OmniVoice TTS voiceover and optionally mux it onto a video")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="Text to synthesize")
    src.add_argument("--text-file", type=Path, help="File containing the script")

    ap.add_argument("-o", "--output", type=Path, required=True, help="Output WAV (or video if --onto)")
    ap.add_argument("--edit-dir", type=Path, default=None, help="Resolve a relative --output under this dir")
    ap.add_argument("--model", type=str, default="k2-fsa/OmniVoice", help="OmniVoice model id or checkpoint path")

    # Voice modes
    ap.add_argument("--instruct", type=str, default=None, help="Voice design, e.g. 'female, british accent'")
    ap.add_argument("--ref-audio", type=str, default=None, help="Reference audio for voice cloning")
    ap.add_argument("--ref-text", type=str, default=None, help="Transcript of the reference audio")
    ap.add_argument("--language", type=str, default=None, help="Language name or code")

    # Generation
    ap.add_argument("--duration", type=float, default=None, help="Fixed output duration in seconds")
    ap.add_argument("--speed", type=float, default=None, help="Speaking-rate factor (>1 faster)")
    ap.add_argument("--device", type=str, default=None, help="cuda / mps / cpu (auto if omitted)")
    ap.add_argument("--dry-run", action="store_true", help="Print the OmniVoice command and exit")

    # Mux onto a video
    ap.add_argument("--onto", type=Path, default=None, help="Lay the narration onto this video; --output becomes the video out path")
    ap.add_argument("--mix", action="store_true", help="Keep original audio ducked under the VO (else replace)")
    ap.add_argument("--duck", type=float, default=0.15, help="Original-audio level under the VO when --mix (0..1)")
    ap.add_argument("--vo-gain", type=float, default=1.0, help="Voiceover gain")
    ap.add_argument("--offset", type=float, default=0.0, help="Delay the VO start (seconds)")
    args = ap.parse_args()

    text = read_text(args)
    if not text:
        sys.exit("no text to synthesize (use --text or --text-file)")

    out = args.output
    if not out.is_absolute() and args.edit_dir:
        out = (args.edit_dir / out).resolve()

    infer_bin = resolve_omnivoice_infer()

    if args.onto:
        # Generate to a sibling WAV, then mux onto the video.
        wav = out.with_suffix(".vo.wav")
        generate_narration(
            infer_bin, text, wav,
            model=args.model, instruct=args.instruct,
            ref_audio=args.ref_audio, ref_text=args.ref_text,
            language=args.language, duration=args.duration, speed=args.speed,
            device=args.device, dry_run=args.dry_run,
        )
        if args.dry_run:
            return
        mux_onto_video(
            args.onto.resolve(), wav, out,
            mix=args.mix, duck=args.duck, vo_gain=args.vo_gain, offset=args.offset,
        )
    else:
        generate_narration(
            infer_bin, text, out,
            model=args.model, instruct=args.instruct,
            ref_audio=args.ref_audio, ref_text=args.ref_text,
            language=args.language, duration=args.duration, speed=args.speed,
            device=args.device, dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
