<p align="center">
  <img src="static/video-use-banner.png" alt="video-use" width="100%">
</p>

# video-use

Introducing **video-use** — edit videos with Claude Code. 100% open source.

> Fork of [browser-use/video-use](https://github.com/browser-use/video-use) with transcription swapped to **local [WhisperX](https://github.com/m-bain/whisperX)** — runs fully offline, no API key required.

Drop raw footage in a folder, chat with Claude Code, get `final.mp4` back. Works for any content — talking heads, montages, tutorials, travel, interviews, and silent process/timelapse videos — without presets or menus.

## What it does

- **Tightens pacing** — cuts dead space between takes and snaps every edit to a word boundary
- **Auto color grades** every segment (warm cinematic, neutral punch, or any custom ffmpeg chain)
- **30ms audio fades** at every cut so you never hear a pop
- **Burns subtitles** in your style — 2-word UPPERCASE chunks by default, fully customizable
- **Generates animation overlays** via [HyperFrames](https://github.com/heygen-com/hyperframes), [Remotion](https://www.remotion.dev/), [Manim](https://www.manim.community/), or PIL — spawned in parallel sub-agents, one per animation
- **Generates AI voiceover/narration** via [OmniVoice](https://github.com/k2-fsa/OmniVoice) TTS (optional) — auto voice, voice design, or voice cloning, then muxed onto any clip (great for narrating silent process/timelapse videos)
- **Self-evaluates the rendered output** at every cut boundary before showing you anything
- **Persists session memory** in `project.md` so next week's session picks up where you left off

## Setup prompt

Paste into Claude Code, Codex, Hermes, Openclaw, or any agent with shell access:

```text
Set up https://github.com/xmllist/video-use for me.

Read install.md first to install this repo, wire up ffmpeg, and register the skill with whichever agent you're running under. Transcription is local (WhisperX) — no API key needed; only ask me for a HuggingFace token if I want speaker diarization. Then read SKILL.md for daily usage, and always read helpers/ because that's where the editing scripts live. After install, don't transcribe anything on your own — just tell me it's ready and wait for me to drop footage into a folder.
```

The agent handles the clone, dependencies (WhisperX + ffmpeg), and skill registration. No API key is needed — transcription runs locally on your machine. Speaker diarization is optional and needs a free [HuggingFace token](https://huggingface.co/settings/tokens).

Then point your agent at a folder of raw takes:

```bash
cd /path/to/your/videos
claude    # or codex, hermes, etc.
```

For always-on editing from your own VPS or Telegram, run the agent through [Browser Use Box](https://browser-use.com/bux). [Watch the 15-second demo](https://www.tiktok.com/@browser_use/video/7639824093721758989).

And in the session:

> edit these into a launch video

It inventories the sources, proposes a strategy, waits for your OK, then produces `edit/final.mp4` next to your sources. All outputs live in `<videos_dir>/edit/` — the skill directory stays clean.

## Manual install

If you'd rather do it by hand:

```bash
# 1. Clone and symlink into your agent's skills directory
git clone https://github.com/xmllist/video-use ~/Developer/video-use
ln -sfn ~/Developer/video-use ~/.claude/skills/video-use        # Claude Code
# ln -sfn ~/Developer/video-use ~/.codex/skills/video-use       # Codex

# 2. Install deps (uv sync pulls WhisperX + torch — first run is a large download)
cd ~/Developer/video-use
uv sync                         # or: pip install -e .
brew install ffmpeg             # required
brew install yt-dlp             # optional, for downloading online sources

# 3. (Optional) speaker diarization — local transcription itself needs no key
cp .env.example .env
$EDITOR .env                    # HF_TOKEN=...  (only for speaker labels via pyannote)
```

## How it works

The LLM never watches the video. It **reads** it — through two layers that together give it everything it needs to cut with word-boundary precision.

<p align="center">
  <img src="static/timeline-view.svg" alt="timeline_view composite — filmstrip + speaker track + waveform + word labels + silence-gap cut candidates" width="100%">
</p>

**Layer 1 — Audio transcript (always loaded).** One local [WhisperX](https://github.com/m-bain/whisperX) pass per source gives word-level timestamps via forced alignment, plus optional speaker diarization (with a HuggingFace token). All takes pack into a single ~12KB `takes_packed.md` — the LLM's primary reading view. It runs offline; unlike hosted ASR it doesn't tag fillers or audio events, so cuts lean on silence gaps and word boundaries. For silent process/timelapse videos (no speech), the cut is driven by visual progress instead.

```
## C0103  (duration: 43.0s, 8 phrases)
  [002.52-005.36] S0 Ninety percent of what a web agent does is completely wasted.
  [006.08-006.74] S0 We fixed this.
```

**Layer 2 — Visual composite (on demand).** `timeline_view` produces a filmstrip + waveform + word labels PNG for any time range. Called only at decision points — ambiguous pauses, retake comparisons, cut-point sanity checks.

> Naive approach: 30,000 frames × 1,500 tokens = **45M tokens of noise**.
> Video Use: **12KB text + a handful of PNGs**.

Same idea as browser-use giving an LLM a structured DOM instead of a screenshot — but for video.

## Pipeline

```
Transcribe ──> Pack ──> LLM Reasons ──> EDL ──> Render ──> Self-Eval
                                                              │
                                                              └─ issue? fix + re-render (max 3)
```

The self-eval loop runs `timeline_view` on the _rendered output_ at every cut boundary — catches visual jumps, audio pops, hidden subtitles. You see the preview only after it passes.

## Design principles

1. **Text + on-demand visuals.** No frame-dumping. The transcript is the surface.
2. **Audio is primary, visuals follow.** Cuts come from speech boundaries and silence gaps.
3. **Ask → confirm → execute → self-eval → persist.** Never touch the cut without strategy approval.
4. **Zero assumptions about content type.** Look, ask, then edit.
5. **12 hard rules, artistic freedom elsewhere.** Production-correctness is non-negotiable. Taste isn't.

See [`SKILL.md`](./SKILL.md) for the full production rules and editing craft.
