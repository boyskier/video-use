#!/bin/bash
# Claude Code on the web: install deps and fetch the local transcription models
# so transcribe.py works offline with no API key.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

python3 -m pip install -q --root-user-action=ignore -e . pytest

# SenseVoice + Silero VAD come from GitHub releases (Hugging Face is often
# blocked in web sandboxes). Cached under ~/.cache/video-use/models.
python3 helpers/transcribe_local.py --download-only

command -v ffmpeg >/dev/null || echo "warning: ffmpeg not found" >&2
