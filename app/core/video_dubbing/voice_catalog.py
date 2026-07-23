from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.tts.base import BaseTTSEngine
from app.tts.engine_registry import create_tts_engine


_logger = logging.getLogger("video_dubbing.voices")


@dataclass(frozen=True)
class VoiceDescriptor:
    voice_id: str
    display_name: str
    language: str | None = None
    gender: str | None = None
    installed: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class VoiceCatalogService:
    """Lists available voices per TTS engine and resolves a ``voice_config``.

    Reuses the existing per-engine voice managers and the Piper voice manager.
    Engines without a listable catalog fall back to an editable single value.
    The catalog never creates a parallel voice system.
    """

    def __init__(self, piper_path: str = "") -> None:
        self.piper_path = piper_path

    def list_voices(self, engine_id: str) -> list[VoiceDescriptor]:
        try:
            if engine_id == "piper":
                return self._list_piper()
            if engine_id in {"kokoro", "kokoro_python"}:
                return self._list_kokoro()
            if engine_id == "qwen":
                return self._list_qwen()
            if engine_id == "chatterbox":
                return self._list_reference_voices("chatterbox")
            if engine_id == "omnivoice":
                return self._list_reference_voices("omnivoice")
            if engine_id in {"openai", "elevenlabs", "gemini", "azure"}:
                return []
            if engine_id.startswith("custom:"):
                return []
        except Exception as exc:  # pragma: no cover - depends on optional installs
            _logger.warning("Voice catalog for %s failed: %s", engine_id, exc)
            return []
        return []

    def resolve_voice_config(
        self,
        engine_id: str,
        voice_id: str,
        base_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = dict(base_config or {})
        config["engine"] = engine_id
        if engine_id == "piper":
            info = self._piper_voice_by_id(voice_id)
            if info is not None:
                merged = info.as_config(1.0)
                merged["engine"] = "piper"
                merged.update(
                    {k: v for k, v in config.items() if k not in ("voice", "model_path", "config_path")}
                )
                return merged
            config["voice"] = voice_id
        elif engine_id in {"kokoro", "kokoro_python"}:
            config["voice"] = voice_id
            config.setdefault("lang", "")
        elif engine_id == "qwen":
            config["speaker"] = voice_id
        elif engine_id in {"chatterbox", "omnivoice"}:
            ref = self._resolve_reference_voice(engine_id, voice_id)
            if ref is not None:
                manager, voice = ref
                try:
                    audio_path = manager.ensure_voice_audio(voice)
                except Exception:
                    audio_path = None
                if audio_path:
                    config["reference_audio_path"] = str(audio_path)
                    if getattr(voice, "ref_text", ""):
                        config["reference_text"] = voice.ref_text
                config.setdefault("reference_audio_path", voice_id)
            else:
                config.setdefault("reference_audio_path", voice_id)
        elif engine_id in {"openai", "gemini", "azure"}:
            config["voice"] = voice_id
        elif engine_id == "elevenlabs":
            config["voice_id"] = voice_id
        return config

    def create_engine(
        self,
        engine_id: str,
        piper_path: str | None = None,
    ) -> BaseTTSEngine:
        from pathlib import Path

        return create_tts_engine(engine_id, Path(piper_path or self.piper_path))

    # ------------------------------------------------------------------ sources

    def _voice_manager(self):
        from pathlib import Path

        from app.utils.paths import application_root
        from app.tts.voice_manager import VoiceManager

        voices_root = Path(self.piper_path).parent if self.piper_path else application_root() / "voices"
        # piper_path points at the piper executable; voices live next to it.
        if self.piper_path:
            candidate = Path(self.piper_path).resolve()
            voices_root = candidate.parent if candidate.is_file() else candidate
        return VoiceManager(voices_root)

    def _list_piper(self) -> list[VoiceDescriptor]:
        result: list[VoiceDescriptor] = []
        try:
            voices = self._voice_manager().discover()
        except Exception:  # pragma: no cover - depends on installed voices
            voices = []
        for voice in voices:
            result.append(
                VoiceDescriptor(
                    voice_id=voice.voice_id,
                    display_name=voice.display_name,
                    language=voice.language or None,
                    installed=True,
                    metadata={
                        "model_path": str(voice.model_path),
                        "config_path": str(voice.config_path),
                    },
                )
            )
        result.sort(key=lambda v: v.display_name.lower())
        return result

    def _piper_voice_by_id(self, voice_id: str):
        try:
            voices = self._voice_manager().discover()
        except Exception:  # pragma: no cover
            voices = []
        for voice in voices:
            if voice.voice_id == voice_id:
                return voice
        return None

    def _list_kokoro(self) -> list[VoiceDescriptor]:
        from app.tts.kokoro_python_manager import KokoroPythonManager

        result: list[VoiceDescriptor] = []
        for voice in KokoroPythonManager().list_voices():
            result.append(
                VoiceDescriptor(
                    voice_id=voice.voice_id,
                    display_name=getattr(voice, "display_name", voice.voice_id),
                    language=getattr(voice, "language", None),
                )
            )
        result.sort(key=lambda v: v.display_name.lower())
        return result

    def _list_qwen(self) -> list[VoiceDescriptor]:
        from app.tts.qwen_manager import QwenManager

        result: list[VoiceDescriptor] = []
        for voice in QwenManager().list_voices():
            result.append(
                VoiceDescriptor(
                    voice_id=voice.voice_id,
                    display_name=getattr(voice, "display_name", voice.voice_id),
                    language=getattr(voice, "language", None),
                )
            )
        result.sort(key=lambda v: v.display_name.lower())
        return result

    def _resolve_reference_voice(self, engine_id: str, voice_id: str):
        try:
            from app.tts.voice_gallery_manager import VoiceGalleryManager

            manager = VoiceGalleryManager()
            manager.ensure_seed_loaded()
            for voice in manager.list_voices(engine_id):
                if voice.name == voice_id and manager.preview_source(voice):
                    return manager, voice
        except Exception:  # pragma: no cover
            return None
        return None

    def _list_reference_voices(self, engine_id: str) -> list[VoiceDescriptor]:
        try:
            from app.tts.voice_gallery_manager import VoiceGalleryManager

            manager = VoiceGalleryManager()
            manager.ensure_seed_loaded()
            voices = [v for v in manager.list_voices(engine_id) if manager.preview_source(v)]
        except Exception as exc:  # pragma: no cover - depends on gallery sync
            _logger.warning("%s gallery voice list failed: %s", engine_id, exc)
            return []
        result: list[VoiceDescriptor] = []
        for voice in voices:
            result.append(
                VoiceDescriptor(
                    voice_id=voice.name,
                    display_name=f"{voice.name}" + (f" — {voice.language}" if voice.language else ""),
                    language=voice.language or None,
                    installed=True,
                    metadata={"engine": engine_id, "has_ref_text": bool(voice.ref_text)},
                )
            )
        result.sort(key=lambda v: v.display_name.lower())
        return result

    def _list_simple(self, _keys: list[str], engine_id: str) -> list[VoiceDescriptor]:
        # Reference-audio based engines: no fixed catalog; the UI keeps the
        # combo editable so the user can paste a path.
        _ = engine_id
        return []
