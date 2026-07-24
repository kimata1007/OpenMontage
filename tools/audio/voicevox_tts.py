"""VOICEVOX local text-to-speech provider tool.

VOICEVOX (https://voicevox.hiroshiba.jp/) is a free Japanese voice synthesis
engine distributed as a local HTTP server. Unlike the other TTS providers in
this package it only speaks Japanese, but it offers many distinct character
voices at zero cost and fully offline once the engine is running.
"""

from __future__ import annotations

import os
import time
import wave
from pathlib import Path
from typing import Any

import requests

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

DEFAULT_SPEAKER_NAME = "青山龍星"
DEFAULT_STYLE_NAME = "ノーマル"


def _engine_url() -> str:
    """Resolve the local VOICEVOX engine base URL.

    Mirrors the env vars already used by explainer-studio-internal's
    start-voicevox.sh / synthesize.mjs (VOICEVOX_PORT), plus a full-URL
    override for engines running on a non-default host.
    """
    override = os.environ.get("VOICEVOX_ENGINE_URL")
    if override:
        return override.rstrip("/")
    port = os.environ.get("VOICEVOX_PORT", "50021")
    return f"http://127.0.0.1:{port}"


class VoicevoxTTS(BaseTool):
    name = "voicevox_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "voicevox"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies = []
    install_instructions = (
        "Start a local VOICEVOX engine before using this tool:\n"
        "  1. Download voicevox_engine from\n"
        "     https://github.com/VOICEVOX/voicevox_engine/releases\n"
        "     (macOS: the *-macos-arm64 or *-macos-x64 asset)\n"
        "  2. Extract it and run: ./run --host 127.0.0.1 --port 50021\n"
        "  3. Wait for `curl http://127.0.0.1:50021/version` to respond.\n"
        "No API key is required. Override the engine location with\n"
        "VOICEVOX_ENGINE_URL (full URL) or VOICEVOX_PORT (port only) if it\n"
        "is not on the default host/port."
    )
    fallback_tools = ["google_tts"]
    agent_skills = []

    capabilities = [
        "text_to_speech",
        "offline_generation",
        "japanese_narration",
    ]
    supports = {
        "voice_cloning": False,
        "multilingual": False,
        "offline": True,
        "native_audio": True,
        "japanese": True,
    }
    best_for = [
        "Japanese-language narration with distinct character voices",
        "free, fully offline TTS once the engine is running",
        "privacy-sensitive local-only workflows",
    ]
    not_good_for = [
        "non-Japanese narration",
        "voice cloning",
    ]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string"},
            "speaker_name": {
                "type": "string",
                "default": DEFAULT_SPEAKER_NAME,
                "description": "VOICEVOX speaker name (character), e.g. '青山龍星'. Ignored when speaker_id is set.",
            },
            "style_name": {
                "type": "string",
                "default": DEFAULT_STYLE_NAME,
                "description": "Style name within the speaker, e.g. 'ノーマル'. Ignored when speaker_id is set.",
            },
            "speaker_id": {
                "type": "integer",
                "description": "VOICEVOX style id, if already known. Overrides speaker_name/style_name lookup.",
            },
            "speed": {
                "type": "number",
                "default": 1.0,
                "description": "speedScale passed to the VOICEVOX audio_query.",
            },
            "post_phoneme_length": {
                "type": "number",
                "default": 0.4,
                "description": "Trailing silence in seconds appended after speech.",
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=128, vram_mb=0, disk_mb=50, network_required=False
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["timeout"])
    idempotency_key_fields = [
        "text", "speaker_id", "speaker_name", "style_name", "speed", "post_phoneme_length",
    ]
    side_effects = ["writes audio file to output_path", "calls the local VOICEVOX engine over HTTP"]
    user_visible_verification = [
        "Listen to generated audio for correct speaker/style and natural Japanese speech",
    ]

    def get_status(self) -> ToolStatus:
        try:
            response = requests.get(f"{_engine_url()}/version", timeout=2.0)
            response.raise_for_status()
            return ToolStatus.AVAILABLE
        except requests.RequestException:
            return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(success=False, error="VOICEVOX engine not reachable. " + self.install_instructions)

        start = time.time()
        try:
            result = self._generate(inputs)
        except requests.RequestException as exc:
            return ToolResult(success=False, error=f"VOICEVOX engine request failed: {exc}")
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))

        result.duration_seconds = round(time.time() - start, 2)
        return result

    def _resolve_speaker(self, inputs: dict[str, Any]) -> tuple[int, str, str]:
        if inputs.get("speaker_id") is not None:
            return int(inputs["speaker_id"]), "", ""

        speaker_name = inputs.get("speaker_name", DEFAULT_SPEAKER_NAME)
        style_name = inputs.get("style_name", DEFAULT_STYLE_NAME)

        response = requests.get(f"{_engine_url()}/speakers", timeout=10.0)
        response.raise_for_status()
        speakers = response.json()

        speaker = next((s for s in speakers if s["name"] == speaker_name), None)
        if speaker is None:
            names = ", ".join(s["name"] for s in speakers)
            raise ValueError(f"Speaker {speaker_name!r} not found. Available: {names}")
        style = next(
            (s for s in speaker["styles"] if s["name"] == style_name),
            speaker["styles"][0],
        )
        return style["id"], speaker_name, style["name"]

    def _generate(self, inputs: dict[str, Any]) -> ToolResult:
        engine = _engine_url()
        speaker_id, speaker_name, style_name = self._resolve_speaker(inputs)
        text = inputs["text"]

        query_response = requests.post(
            f"{engine}/audio_query",
            params={"speaker": speaker_id, "text": text},
            timeout=30.0,
        )
        query_response.raise_for_status()
        query = query_response.json()
        query["speedScale"] = inputs.get("speed", 1.0)
        query["postPhonemeLength"] = inputs.get("post_phoneme_length", 0.4)

        synthesis_response = requests.post(
            f"{engine}/synthesis",
            params={"speaker": speaker_id},
            json=query,
            timeout=60.0,
        )
        synthesis_response.raise_for_status()

        output_path = Path(inputs.get("output_path", "voicevox_output.wav"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(synthesis_response.content)

        with wave.open(str(output_path), "rb") as wav_file:
            duration = wav_file.getnframes() / wav_file.getframerate()

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "style_name": style_name,
                "text_length": len(text),
                "output": str(output_path),
                "format": "wav",
                "audio_duration_seconds": round(duration, 3),
            },
            artifacts=[str(output_path)],
        )
