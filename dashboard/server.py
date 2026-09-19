"""FastAPI dashboard — config, lifecycle, and test endpoints."""
from __future__ import annotations

import asyncio
import json
import os
import random
import secrets
import socket
import sys
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response

from audio.player import list_output_devices
from config import (
    BASE_DIR,
    AppConfig,
    Secrets,
    activate_profile,
    clone_profile,
    delete_profile,
    list_profiles,
    load_profile,
    save_profile,
)
from core import Orchestrator, Persona
from llm import build_provider

# production/ has moved to the sibling lets-watch repo; it still imports
# wallie-V2's shared tts/llm/imagegen infra, and this dashboard still hosts
# its Stage 7 routes pending real extraction (see docs/overlay-host-v3-handoff.md
# in lets-watch). Both repos are expected to be checked out as siblings on disk.
_LETS_WATCH = Path(__file__).resolve().parent.parent.parent / "lets-watch"
if _LETS_WATCH.is_dir() and str(_LETS_WATCH) not in sys.path:
    sys.path.insert(0, str(_LETS_WATCH))

from production.service import (
    ProductionService,
    SpeechProviderBlocker,
)
from production.workflow import (
    ProductionWorkflow,
    WorkflowBlocked,
    WorkflowError,
)
from tts import build_tts

STATIC_DIR = Path(__file__).parent / "static"
PRODUCTION_ROOT = BASE_DIR / "projects"


# -------------------------------------------------------------------
# PIN authentication (closure-based, shared state)
# -------------------------------------------------------------------
_PUBLIC_PATHS = frozenset({"/login", "/api/auth/login"})


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


_DEFAULT_TEST_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.1-8b-instant",
    "openrouter": "openai/gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.2",
}


def _default_model_for(provider: str) -> str:
    return _DEFAULT_TEST_MODELS.get(provider, "")


import re as _re

_KEY_PATTERNS = [
    _re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    _re.compile(r"sk-or-[A-Za-z0-9_\-]{10,}"),
    _re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    _re.compile(r"gsk_[A-Za-z0-9_\-]{20,}"),
    _re.compile(r"AIza[A-Za-z0-9_\-]{20,}"),       # Google API keys
    _re.compile(r"oauth:[A-Za-z0-9_\-]{20,}"),     # Twitch
    _re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}"),
    _re.compile(r"\b[A-Fa-f0-9]{40,}\b"),          # generic hex tokens
]


def _scrub_error(msg: str, limit: int = 220) -> str:
    out = msg
    for pat in _KEY_PATTERNS:
        out = pat.sub("[redacted]", out)
    return out[:limit]


class DashboardState:
    def __init__(self) -> None:
        self.orchestrator: Optional[Orchestrator] = None
        self.clients: set[WebSocket] = set()
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)

    def attach_logger(self) -> None:
        def sink(message) -> None:
            record = message.record
            entry = {
                "type": "log",
                "level": record["level"].name,
                "time": record["time"].isoformat(),
                "msg": record["message"],
            }
            try:
                self._queue.put_nowait(entry)
            except asyncio.QueueFull:
                pass
        logger.add(sink, level="INFO", enqueue=False)

    async def broadcaster(self) -> None:
        while True:
            entry = await self._queue.get()
            dead: list[WebSocket] = []
            for ws in list(self.clients):
                try:
                    await ws.send_text(json.dumps(entry))
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)


class ProfileCreateBody(BaseModel):
    name: str
    clone_from: Optional[str] = None


class TestPersonaBody(BaseModel):
    kind: str = "monologue"  # monologue | chat | vision
    topic: Optional[str] = None
    chat_text: Optional[str] = None
    chat_user: Optional[str] = None


class TestVoiceBody(BaseModel):
    text: str


class TestExpressionBody(BaseModel):
    expression: str  # slot ("happy", "hype", ...) OR a raw hotkey id/name


class TestLookBody(BaseModel):
    x: float = 0.0
    y: float = 0.0
    hold_sec: float = 0.6


class SecretUpdateBody(BaseModel):
    env: str
    value: str  # empty string deletes


class SecretBulkBody(BaseModel):
    values: dict[str, str]


class TestProviderBody(BaseModel):
    # "openai" | "groq" | "openrouter" | "anthropic" | "gemini" | "fish" | "elevenlabs" | "piper"
    provider: str


class PinBody(BaseModel):
    pin: str


# -------------------------------------------------------------------
# Production workflow (Stage 7)
# -------------------------------------------------------------------
class ProductionProjectBody(BaseModel):
    project_id: str
    title: str = ""
    source_url: str = ""
    source_platform: str = "other"
    source_language: str = "en"
    commentary_language: str = "en"


class ProductionRightsBody(BaseModel):
    basis: str
    evidence: Optional[list[str]] = None
    notes: Optional[str] = None
    verify: bool = False
    verifier: str = ""
    creator: str = ""
    title: str = ""
    source_url: str = ""
    license_name: str = ""
    license_url: str = ""
    credit_line: str = ""
    attribution_notes: str = ""


class ProductionEvidenceBody(BaseModel):
    name: str
    content: str


class ProductionPathBody(BaseModel):
    path: str
    source_url: str = ""
    notes: str = ""


class ProductionCommentaryBody(BaseModel):
    mode: str = "en"
    style: str = "reaction"
    persona: str = ""
    voice: str = ""
    speech_model: str = ""
    mp3_preview: bool = False


class ProductionApproveBody(BaseModel):
    approver: str
    notes: str = ""


def _production_workflow(project_id: str) -> ProductionWorkflow:
    try:
        return ProductionWorkflow.open(PRODUCTION_ROOT, project_id)
    except FileNotFoundError:
        raise HTTPException(404, f"project not found: {project_id!r}")
    except ValueError as exc:
        raise HTTPException(400, str(exc))


def _build_app(
    state: DashboardState,
    initial: Optional[Orchestrator],
    *,
    pin: str = "",
) -> FastAPI:
    state.orchestrator = initial
    app = FastAPI(title="Wallie Dashboard")

    _pin = pin
    _sessions: set[str] = set()

    def _is_authed(cookies: dict[str, str]) -> bool:
        s = cookies.get("wallie_session")
        return bool(s and s in _sessions)

    if _pin:
        @app.middleware("http")
        async def _pin_gate(request: Request, call_next: Any) -> Response:
            path = request.url.path
            if path in _PUBLIC_PATHS or path == "/static/style.css":
                return await call_next(request)
            if _is_authed(request.cookies):
                return await call_next(request)
            if path.startswith("/api/"):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return RedirectResponse("/login", status_code=302)

    @app.on_event("startup")
    async def _on_start() -> None:
        state.attach_logger()
        asyncio.create_task(state.broadcaster(), name="dash-broadcast")

    # ---------- auth ----------
    @app.get("/login")
    def login_page() -> FileResponse:
        if not _pin:
            return RedirectResponse("/", status_code=302)  # type: ignore[return-value]
        return FileResponse(str(STATIC_DIR / "login.html"))

    @app.post("/api/auth/login")
    def api_auth_login(body: PinBody) -> Any:
        if not _pin:
            return {"ok": True}
        if not secrets.compare_digest(body.pin, _pin):
            raise HTTPException(403, "wrong pin")
        token = secrets.token_urlsafe(32)
        _sessions.add(token)
        response = JSONResponse({"ok": True})
        response.set_cookie(
            "wallie_session", token,
            httponly=True, samesite="lax", max_age=86400,
        )
        return response

    # ---------- profiles ----------
    @app.get("/api/profiles")
    def api_profiles() -> dict[str, Any]:
        cfg = load_profile()
        return {"active": cfg.profile_name, "profiles": list_profiles() or [cfg.profile_name]}

    @app.post("/api/profiles")
    def api_profiles_create(body: ProfileCreateBody) -> dict[str, Any]:
        name = body.name.strip()
        if not name:
            raise HTTPException(400, "name required")
        if body.clone_from:
            clone_profile(body.clone_from, name)
        else:
            cfg = AppConfig(profile_name=name)
            save_profile(cfg, name)
        activate_profile(name)
        return {"ok": True, "active": name}

    @app.put("/api/profiles/{name}/activate")
    def api_profiles_activate(name: str) -> dict[str, Any]:
        cfg = activate_profile(name)
        return {"ok": True, "active": cfg.profile_name}

    @app.delete("/api/profiles/{name}")
    def api_profiles_delete(name: str) -> dict[str, Any]:
        ok = delete_profile(name)
        return {"ok": ok}

    # ---------- config ----------
    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        return load_profile().model_dump()

    @app.put("/api/config")
    async def put_config(payload: dict[str, Any]) -> dict[str, Any]:
        # Merge onto the existing profile so any config section the UI doesn't send
        # (e.g. hearing) is preserved instead of being silently reset to defaults.
        existing = load_profile().model_dump()
        existing.update(payload)
        cfg = AppConfig(**existing)
        save_profile(cfg, cfg.profile_name)
        return {"ok": True}

    # ---------- orchestrator lifecycle ----------
    @app.get("/api/status")
    def get_status() -> dict[str, Any]:
        orch = state.orchestrator
        snap: dict[str, Any] = orch.status() if orch else {"running": False}
        snap["profile"] = load_profile().profile_name
        return snap

    @app.get("/api/audio-devices")
    def audio_devices() -> list[dict[str, Any]]:
        return list_output_devices()

    @app.post("/api/start")
    async def api_start() -> dict[str, Any]:
        from wallie import build_orchestrator
        if state.orchestrator and state.orchestrator.status().get("running"):
            return {"ok": True, "already": True}
        state.orchestrator = build_orchestrator()
        await state.orchestrator.start()
        return {"ok": True}

    @app.post("/api/stop")
    async def api_stop() -> dict[str, Any]:
        if state.orchestrator:
            await state.orchestrator.stop()
        return {"ok": True}

    @app.post("/api/break")
    async def api_break() -> dict[str, Any]:
        orch = state.orchestrator
        if not orch or not orch.status().get("running"):
            raise HTTPException(400, "Orchestrator not running")
        orch.trigger_break()
        return {"ok": True}

    @app.post("/api/resume")
    async def api_resume() -> dict[str, Any]:
        orch = state.orchestrator
        if not orch or not orch.status().get("running"):
            raise HTTPException(400, "Orchestrator not running")
        orch.resume_from_break()
        return {"ok": True}

    # ---------- Minecraft Play mode ----------
    _REPO = Path(__file__).resolve().parent.parent

    @app.post("/api/play/install")
    async def api_play_install() -> dict[str, Any]:
        import asyncio as _aio
        proc = await _aio.create_subprocess_exec(
            sys.executable, str(_REPO / "scripts" / "install_minecraft.py"),
            cwd=str(_REPO), stdout=_aio.subprocess.PIPE, stderr=_aio.subprocess.STDOUT,
        )
        try:
            out, _ = await _aio.wait_for(proc.communicate(), timeout=300)
        except _aio.TimeoutError:
            proc.kill()
            raise HTTPException(504, "Install timed out — check your connection and try again")
        log = (out or b"").decode("utf-8", "ignore")
        return {"ok": proc.returncode == 0, "log": log[-4000:]}

    @app.post("/api/play/launch")
    async def api_play_launch() -> dict[str, Any]:
        import asyncio as _aio
        cfg = load_profile()
        if not cfg.play.enabled:
            raise HTTPException(400, "Play mode is OFF — enable it in the Play section first")
        if state.orchestrator and state.orchestrator.status().get("running"):
            await state.orchestrator.stop()        # the live runner starts its own orchestrator
            state.orchestrator = None
        await _aio.create_subprocess_exec(
            sys.executable, str(_REPO / "scripts" / "run_wallie_live.py"),
            cfg.play.goal, cfg.profile_name, cwd=str(_REPO),
        )
        return {"ok": True, "msg": "Wallie Play launching — focus the Minecraft window. F8 stops it."}

    # ---------- memory endpoints ----------
    @app.get("/api/memory")
    def api_memory_get() -> dict[str, Any]:
        from config import PROFILES_DIR
        from core import MemoryStore
        cfg = load_profile()
        profile_name = cfg.profile_name or "default"
        store = MemoryStore(PROFILES_DIR / f"{profile_name}.memory.json")
        store.load()
        return {
            "notes": store.notes,
            "viewer_log": store.recent_viewers(100),
            "profile": profile_name,
        }

    @app.delete("/api/memory")
    def api_memory_clear() -> dict[str, Any]:
        from config import PROFILES_DIR
        from core import MemoryStore
        cfg = load_profile()
        profile_name = cfg.profile_name or "default"
        path = PROFILES_DIR / f"{profile_name}.memory.json"
        if path.exists():
            path.unlink()
        return {"ok": True, "cleared": profile_name}

    # ---------- test endpoints ----------
    @app.post("/api/test/persona")
    async def test_persona(body: TestPersonaBody) -> dict[str, Any]:
        cfg = load_profile()
        persona = Persona.from_config(cfg.persona)
        llm = build_provider(cfg.llm, Secrets())
        system = persona.system_prompt(
            topic=body.topic or (cfg.topics.topics[0] if cfg.topics.topics else None),
            vision_enabled=cfg.vision.enabled,
            topic_drift_style=cfg.topics.drift_style,
        )
        if body.kind == "chat":
            user = persona.chat_turn(
                username=body.chat_user or "regular_viewer",
                platform="twitch",
                text=body.chat_text or "hey what's up today",
                is_highlight=False,
            )
        elif body.kind == "vision":
            user = persona.vision_turn()
        else:
            oc = cfg.orchestrator
            user = persona.monologue_turn(
                topic=body.topic,
                sentences_min=oc.segment_sentences_min,
                sentences_max=oc.segment_sentences_max,
                topic_drift_style=cfg.topics.drift_style,
            )
        try:
            out: list[str] = []
            async for token in llm.stream(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=cfg.llm.temperature,
                top_p=cfg.llm.top_p,
                max_tokens=min(cfg.llm.max_tokens, 200),
                presence_penalty=cfg.llm.presence_penalty,
                frequency_penalty=cfg.llm.frequency_penalty,
            ):
                out.append(token)
            return {"ok": True, "text": "".join(out).strip(), "system_preview": system}
        finally:
            await llm.aclose()

    # ---------- avatar ----------
    def _live_avatar():
        orch = state.orchestrator
        avatar = getattr(orch, "_avatar", None) if orch else None
        if avatar is None:
            raise HTTPException(400, "Avatar not enabled. Start the orchestrator with avatar.enabled=true.")
        return avatar

    @app.get("/api/avatar/status")
    async def avatar_status() -> dict[str, Any]:
        orch = state.orchestrator
        avatar = getattr(orch, "_avatar", None) if orch else None
        if avatar is None:
            return {"enabled": False, "connected": False}
        return avatar.status()

    @app.get("/api/avatar/hotkeys")
    async def avatar_hotkeys() -> dict[str, Any]:
        avatar = _live_avatar()
        hotkeys = await avatar.query_hotkeys()
        return {"hotkeys": hotkeys}

    @app.get("/api/avatar/model")
    async def avatar_model() -> dict[str, Any]:
        avatar = _live_avatar()
        info = await avatar.query_model_info()
        return {"model": info}

    @app.post("/api/test/expression")
    async def test_expression(body: TestExpressionBody) -> dict[str, Any]:
        avatar = _live_avatar()
        expression = body.expression.strip()
        if not expression:
            raise HTTPException(400, "expression required")
        # Try the slot path first ("happy", "hype" etc.). If nothing matches,
        # fall back to a raw hotkey id/name.
        slot_attr = f"expr_{expression}"
        cfg = load_profile().avatar
        if hasattr(cfg, slot_attr) and getattr(cfg, slot_attr, ""):
            await avatar.trigger_emotion(expression)
        else:
            await avatar.trigger_expression(expression)
        return {"ok": True, "expression": expression}

    @app.post("/api/test/avatar_look")
    async def test_avatar_look(body: TestLookBody) -> dict[str, Any]:
        avatar = _live_avatar()
        await avatar.look_at(body.x, body.y, hold_sec=body.hold_sec)
        return {"ok": True}

    # ---------- secrets (API keys) ----------
    @app.get("/api/secrets")
    def api_secrets_list() -> dict[str, Any]:
        from secrets_store import list_secrets
        return {"secrets": list_secrets()}

    @app.put("/api/secrets")
    def api_secrets_update(body: SecretUpdateBody) -> dict[str, Any]:
        from secrets_store import SECRET_FIELDS, set_secret
        if body.env not in SECRET_FIELDS:
            raise HTTPException(400, "unknown secret field")
        try:
            set_secret(body.env, body.value)
        except Exception as e:
            raise HTTPException(500, f"failed to write secret: {e}")
        return {"ok": True, "env": body.env, "is_set": bool(body.value.strip())}

    @app.post("/api/secrets/bulk")
    def api_secrets_bulk(body: SecretBulkBody) -> dict[str, Any]:
        from secrets_store import update_many
        accepted = update_many(body.values)
        return {"ok": True, "updated": accepted}

    @app.post("/api/secrets/test")
    async def api_secrets_test(body: TestProviderBody) -> dict[str, Any]:
        from config import LLMConfig, Secrets, TTSConfig
        prov = body.provider
        try:
            if prov in ("openai", "groq", "openrouter", "anthropic", "gemini", "ollama"):
                from llm import build_provider
                cfg = LLMConfig(provider=prov, model=_default_model_for(prov), max_tokens=4)
                client = build_provider(cfg, Secrets())
                try:
                    out: list[str] = []
                    async def _drain():
                        async for tok in client.stream(
                            [
                                {"role": "system", "content": "Say only the word ok."},
                                {"role": "user", "content": "ok"},
                            ],
                            temperature=0.0,
                            top_p=1.0,
                            max_tokens=4,
                        ):
                            out.append(tok)
                    await asyncio.wait_for(_drain(), timeout=10.0)
                    return {"ok": True, "preview": "".join(out).strip()[:30]}
                finally:
                    await client.aclose()
            if prov in ("fish", "elevenlabs", "piper"):
                from tts import build_tts
                cfg = TTSConfig(provider=prov)
                if prov == "piper":
                    # Piper needs a model file path; can't test without it.
                    return {"ok": False, "error": "Piper test requires piper_model_path in config; not auto-testable here."}
                client = build_tts(cfg, Secrets())
                try:
                    received = 0
                    async def _drain():
                        nonlocal received
                        async for chunk in client.synthesize("test"):
                            received += len(chunk)
                            if received > 1024:
                                return
                    await asyncio.wait_for(_drain(), timeout=10.0)
                    if received == 0:
                        return {"ok": False, "error": "no audio bytes returned"}
                    return {"ok": True, "preview": f"{received} bytes received"}
                finally:
                    await client.aclose()
            return {"ok": False, "error": f"unknown provider: {prov}"}
        except asyncio.TimeoutError:
            return {"ok": False, "error": "timeout (>10s) — server unreachable or key wrong"}
        except Exception as e:
            return {"ok": False, "error": _scrub_error(str(e))}

    @app.post("/api/audio/reset")
    async def audio_reset() -> dict[str, Any]:
        orch = state.orchestrator
        if orch is None:
            raise HTTPException(400, "Orchestrator not running")
        player = getattr(orch, "_player", None)
        if player is None:
            raise HTTPException(500, "no audio player")
        player.reset()
        return {"ok": True}

    @app.post("/api/test/vision")
    async def test_vision() -> dict[str, Any]:
        cfg = load_profile()
        if not cfg.llm.vision_capable:
            raise HTTPException(
                400,
                "Engine.vision_capable is OFF. Toggle it on for a vision-capable model.",
            )
        # Capture one frame.
        try:
            from vision import ScreenCapture
        except ModuleNotFoundError as e:
            raise HTTPException(500, f"vision deps missing: {e}")
        cap = ScreenCapture(
            monitor_index=cfg.vision.monitor_index,
            max_edge_px=cfg.vision.max_edge_px,
        )
        try:
            frame = await asyncio.to_thread(cap.grab)
        finally:
            cap.close()

        persona = Persona.from_config(cfg.persona)
        system = persona.system_prompt(
            topic=None, vision_enabled=True, session_notes=None,
            topic_drift_style=cfg.topics.drift_style,
        )
        # Force the screen-anchored prompt.
        oc = cfg.orchestrator
        user = persona.vision_turn(
        change_type="scene",
        mood_label="warm",
        target_sentences=1,   
        screen_activity="",
    )
        msgs = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image", "data": frame.jpeg, "mime": "image/jpeg"},
                ],
            },
        ]
        llm = build_provider(cfg.llm, Secrets())
        try:
            tokens: list[str] = []
            async for tok in llm.stream(
                msgs,
                temperature=cfg.llm.temperature,
                top_p=cfg.llm.top_p,
                max_tokens=min(cfg.llm.max_tokens, 80),
                presence_penalty=cfg.llm.presence_penalty,
                frequency_penalty=cfg.llm.frequency_penalty,
            ):
                tokens.append(tok)
        except Exception as e:
            await llm.aclose()
            raise HTTPException(500, f"LLM error: {e}")
        finally:
            await llm.aclose()
        text = "".join(tokens).strip()
        return {
            "ok": True,
            "frame_size": [frame.width, frame.height],
            "frame_bytes": len(frame.jpeg),
            "model": cfg.llm.model,
            "provider": cfg.llm.provider,
            "text": text,
        }

    @app.post("/api/test/voice")
    async def test_voice(body: TestVoiceBody) -> dict[str, Any]:
        cfg = load_profile()
        orch = state.orchestrator
        if orch and orch.status().get("running"):
            player = orch._player  # noqa: SLF001
            tts = build_tts(cfg.tts, Secrets())
            try:
                async for pcm in tts.synthesize(body.text):
                    await player.write(pcm)
            finally:
                await tts.aclose()
            return {"ok": True, "routed": "live-player"}
        from audio import AudioPlayer
        tts = build_tts(cfg.tts, Secrets())
        player = AudioPlayer(sample_rate=tts.sample_rate, channels=tts.channels,
                             device=(cfg.tts.output_device or None))
        player.start()
        try:
            async for pcm in tts.synthesize(body.text):
                await player.write(pcm)
            await asyncio.sleep(0.3)
            await player.wait_drained()
        finally:
            player.close()
            await tts.aclose()
        return {"ok": True, "routed": "preview-player"}

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket) -> None:
        if _pin and not _is_authed(ws.cookies):
            await ws.accept()
            await ws.close(code=4001, reason="unauthorized")
            return
        await ws.accept()
        state.clients.add(ws)
        try:
            snap = state.orchestrator.status() if state.orchestrator else {"running": False}
            await ws.send_text(json.dumps({"type": "status", "data": snap}))
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            state.clients.discard(ws)

    # ---------- production workflow (Stage 7) ----------
    def _production_action(work: ProductionWorkflow, func: Any) -> dict[str, Any]:
        try:
            func(work)
        except SpeechProviderBlocker as exc:
            raise HTTPException(409, exc.to_dict())
        except WorkflowBlocked as exc:
            raise HTTPException(409, exc.to_dict())
        except WorkflowError as exc:
            raise HTTPException(400, _scrub_error(str(exc)))
        except (ValueError, OSError) as exc:
            raise HTTPException(400, _scrub_error(str(exc)))
        except Exception as exc:
            raise HTTPException(500, _scrub_error(str(exc)))
        return work.status()

    @app.get("/api/production/projects")
    def production_projects() -> dict[str, Any]:
        return {"projects": ProductionWorkflow.list_projects(PRODUCTION_ROOT)}

    @app.post("/api/production/projects")
    def production_create(body: ProductionProjectBody) -> dict[str, Any]:
        try:
            ProductionWorkflow.create(
                PRODUCTION_ROOT,
                body.project_id,
                title=body.title,
                source_url=body.source_url,
                source_platform=body.source_platform,
                source_language=body.source_language,
                commentary_language=body.commentary_language,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return _production_workflow(body.project_id).status()

    @app.get("/api/production/projects/{project_id}")
    def production_status(project_id: str) -> dict[str, Any]:
        return _production_workflow(project_id).status()

    @app.put("/api/production/projects/{project_id}/rights")
    def production_rights(project_id: str, body: ProductionRightsBody) -> dict[str, Any]:
        work = _production_workflow(project_id)

        def _apply(w: ProductionWorkflow) -> None:
            if any(
                value
                for value in (
                    body.creator,
                    body.title,
                    body.source_url,
                    body.license_name,
                    body.license_url,
                    body.credit_line,
                    body.attribution_notes,
                )
            ):
                w.update_attribution(
                    creator=body.creator,
                    title=body.title,
                    source_url=body.source_url,
                    license_name=body.license_name,
                    license_url=body.license_url,
                    credit_line=body.credit_line,
                    notes=body.attribution_notes,
                )
            w.update_rights(body.basis, evidence=body.evidence, notes=body.notes)
            if body.verify:
                w.verify_rights(verifier=body.verifier)

        return _production_action(work, _apply)

    @app.post("/api/production/projects/{project_id}/evidence")
    def production_evidence(
        project_id: str, body: ProductionEvidenceBody
    ) -> dict[str, Any]:
        work = _production_workflow(project_id)
        return _production_action(
            work, lambda w: w.store_evidence_file(body.name, body.content)
        )

    @app.post("/api/production/projects/{project_id}/source")
    def production_source(project_id: str, body: ProductionPathBody) -> dict[str, Any]:
        work = _production_workflow(project_id)
        return _production_action(
            work,
            lambda w: w.import_source(
                body.path, source_url=body.source_url, notes=body.notes
            ),
        )

    @app.post("/api/production/projects/{project_id}/caption")
    def production_caption(project_id: str, body: ProductionPathBody) -> dict[str, Any]:
        work = _production_workflow(project_id)
        return _production_action(
            work,
            lambda w: w.import_caption(
                body.path, source_url=body.source_url, notes=body.notes
            ),
        )

    @app.post("/api/production/projects/{project_id}/commentary")
    def production_commentary(
        project_id: str, body: ProductionCommentaryBody
    ) -> dict[str, Any]:
        work = _production_workflow(project_id)
        cfg = load_profile()
        service = ProductionService(
            PRODUCTION_ROOT,
            config=cfg,
            secrets=Secrets(),
            mp3_preview=body.mp3_preview,
        )

        def _apply(w: ProductionWorkflow) -> None:
            w.set_language(body.mode)
            service.produce(
                project_id,
                mode=body.mode,
                style=body.style,
                persona=body.persona,
                voice=body.voice,
                speech_model=body.speech_model,
            )

        return _production_action(work, _apply)

    @app.post("/api/production/projects/{project_id}/review")
    def production_review(project_id: str) -> dict[str, Any]:
        work = _production_workflow(project_id)
        service = ProductionService(
            PRODUCTION_ROOT,
            config=load_profile(),
            secrets=Secrets(),
            mp3_preview=True,
        )
        return _production_action(work, lambda w: service.render_review(project_id))

    @app.post("/api/production/projects/{project_id}/approve")
    def production_approve(
        project_id: str, body: ProductionApproveBody
    ) -> dict[str, Any]:
        work = _production_workflow(project_id)
        return _production_action(
            work, lambda w: w.approve(body.approver, notes=body.notes)
        )

    @app.post("/api/production/projects/{project_id}/export")
    def production_export(project_id: str) -> dict[str, Any]:
        work = _production_workflow(project_id)
        service = ProductionService(
            PRODUCTION_ROOT,
            config=load_profile(),
            secrets=Secrets(),
            mp3_preview=True,
        )
        return _production_action(work, lambda w: service.export_final(project_id))

    # ---------- static UI ----------
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(STATIC_DIR / "index.html"))

    return app


async def serve(orchestrator: Optional[Orchestrator]) -> None:
    state = DashboardState()

    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("DASHBOARD_PORT", "8765"))
    pin = os.getenv("DASHBOARD_PIN", "").strip()
    is_remote = host not in ("127.0.0.1", "localhost", "::1")

    if pin:
        logger.info("dashboard: PIN auth active (from DASHBOARD_PIN)")
    elif is_remote:
        pin = str(random.randint(1000, 9999))
        logger.info(f"dashboard: auto-generated PIN: {pin}")
        logger.info("dashboard: set DASHBOARD_PIN in .env for a permanent PIN")
    else:
        pin = ""

    if is_remote and not pin:
        logger.warning(
            f"dashboard: bound to {host}:{port} WITHOUT PIN protection. "
            "Anyone on your network can access the dashboard."
        )

    app = _build_app(state, orchestrator, pin=pin)

    if is_remote:
        local_ip = _get_local_ip()
        logger.info(f"dashboard: access from your phone/tablet: http://{local_ip}:{port}")

    config = uvicorn.Config(app, host=host, port=port, log_level="info", lifespan="on")
    server = uvicorn.Server(config)
    logger.info(f"dashboard: http://{host}:{port}")
    await server.serve()
