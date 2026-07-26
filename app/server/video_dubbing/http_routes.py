"""HTTP routes for the video-dubbing control plane.

Registered on the EXISTING FastAPI app from :mod:`app.server.http_app`, under
``/api/video-dubbing``. They call the same :class:`VideoDubbingFacade` as the
MCP tools, so there is no duplicated business logic. The existing token guard
already protects these routes.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .errors import DubbingError, DubbingNotFoundError, DubbingProjectBusyError, DubbingProjectChangedError
from .facade import VideoDubbingFacade
from .fault_injection import FaultInjector


def _err(exc: DubbingError) -> HTTPException:
    return HTTPException(status_code=exc.http_status, detail=exc.to_dict())


def _safe(facade: VideoDubbingFacade, fn):  # noqa: ANN001
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except DubbingProjectChangedError as exc:
            raise _err(exc) from exc
        except DubbingProjectBusyError as exc:
            raise _err(exc) from exc
        except DubbingNotFoundError as exc:
            raise _err(exc) from exc
        except DubbingError as exc:
            raise _err(exc) from exc

    return _wrapper


async def _json(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="JSON body expected.") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object expected.")
    return body


def register_video_dubbing_http_routes(
    app: FastAPI,
    facade: VideoDubbingFacade,
    *,
    fault_injector: FaultInjector | None = None,
) -> None:
    fault_injector = fault_injector or FaultInjector()
    prefix = "/api/video-dubbing"

    @app.get(f"{prefix}/projects")
    def http_list_projects() -> list[dict[str, Any]]:
        return facade.list_projects()

    @app.post(f"{prefix}/projects")
    async def http_create_project(request: Request) -> dict[str, Any]:
        body = await _json(request)
        try:
            return facade.create_project(
                str(body.get("project_dir") or ""),
                title=str(body.get("title") or "Video Dubbing"),
                settings=body.get("settings"),
            )
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.post(f"{prefix}/projects/open")
    async def http_open_project(request: Request) -> dict[str, Any]:
        body = await _json(request)
        try:
            return facade.open_project(str(body.get("project_dir") or ""))
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.get(f"{prefix}/projects/{{project_id}}")
    def http_get_project(project_id: str, include_cues: bool = False) -> dict[str, Any]:
        try:
            return facade.get_project(project_id, include_cues=include_cues)
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.get(f"{prefix}/projects/{{project_id}}/state")
    def http_get_state(project_id: str) -> dict[str, Any]:
        try:
            return facade.get_project_state(project_id)
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.delete(f"{prefix}/projects/{{project_id}}")
    def http_delete_project(project_id: str, confirm: bool = False) -> dict[str, Any]:
        try:
            return facade.delete_project(project_id, confirm=confirm)
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.post(f"{prefix}/projects/{{project_id}}/video")
    async def http_attach_video(project_id: str, request: Request) -> dict[str, Any]:
        body = await _json(request)
        try:
            return facade.attach_video(
                project_id,
                str(body.get("video_path") or ""),
                expected_revision=body.get("expected_revision"),
            )
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.post(f"{prefix}/projects/{{project_id}}/srt")
    async def http_import_srt(project_id: str, request: Request) -> dict[str, Any]:
        body = await _json(request)
        try:
            return facade.import_srt(
                project_id,
                str(body.get("srt_path") or ""),
                expected_revision=body.get("expected_revision"),
            )
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.get(f"{prefix}/projects/{{project_id}}/analyze")
    def http_analyze(project_id: str) -> dict[str, Any]:
        try:
            return facade.analyze_project(project_id)
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.get(f"{prefix}/projects/{{project_id}}/cues")
    def http_list_cues(
        project_id: str,
        page: int = 1,
        page_size: int = 50,
        include_text: bool = False,
        status: str | None = None,
        has_overflow: bool | None = None,
        needs_tts: bool | None = None,
    ) -> dict[str, Any]:
        try:
            return facade.list_cues(
                project_id,
                page=page,
                page_size=page_size,
                include_text=include_text,
                status=status,
                has_overflow=has_overflow,
                needs_tts=needs_tts,
            )
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.get(f"{prefix}/projects/{{project_id}}/cues/{{sequence}}")
    def http_get_cue(project_id: str, sequence: int, include_text: bool = True) -> dict[str, Any]:
        try:
            return facade.get_cue(project_id, sequence, include_text=include_text)
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.patch(f"{prefix}/projects/{{project_id}}/cues/{{sequence}}")
    async def http_patch_cue(project_id: str, sequence: int, request: Request) -> dict[str, Any]:
        body = await _json(request)
        try:
            if "text" in body:
                return facade.update_cue_text(
                    project_id, sequence, str(body["text"]), expected_revision=body.get("expected_revision")
                )
            return facade.update_cue_timing(
                project_id,
                sequence,
                int(body.get("start_ms", 0)),
                int(body.get("end_ms", 0)),
                expected_revision=body.get("expected_revision"),
            )
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.post(f"{prefix}/projects/{{project_id}}/simulate")
    async def http_simulate(project_id: str, request: Request) -> dict[str, Any]:
        body = await _json(request)
        try:
            return facade.simulate_timing_changes(
                project_id,
                list(body.get("sequences") or []),
                body.get("changes"),
            )
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.post(f"{prefix}/projects/{{project_id}}/actions")
    async def http_actions(project_id: str, request: Request) -> dict[str, Any]:
        body = await _json(request)
        action = str(body.get("action") or "")
        arguments = dict(body.get("arguments") or {})
        try:
            from .action_registry import validate_action

            validate_action(action)
            method = getattr(facade, action, None)
            if not callable(method):
                raise HTTPException(status_code=404, detail=f"action not implemented: {action}")
            arguments.pop("project_id", None)
            if body.get("dry_run"):
                return {"action": action, "dry_run": True, "arguments": arguments}
            result = method(project_id=project_id, **arguments)
            return result if isinstance(result, dict) else {"value": result}
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.post(f"{prefix}/projects/{{project_id}}/jobs")
    async def http_create_job(project_id: str, request: Request) -> dict[str, Any]:
        body = await _json(request)
        job_type = str(body.get("job_type") or body.get("action") or "")
        sequences = body.get("sequences")
        try:
            mapping = {
                "generate_cues": lambda: facade.generate_cues(
                    project_id, list(sequences or []), force=bool(body.get("force", False)),
                    error_policy=str(body.get("error_policy", "continue")), wait=bool(body.get("wait", False))
                ),
                "generate_missing": lambda: facade.generate_missing(project_id, wait=bool(body.get("wait", False))),
                "generate_all": lambda: facade.generate_all(
                    project_id, force=bool(body.get("force", False)), wait=bool(body.get("wait", False))
                ),
                "refit_cues": lambda: facade.refit_cues(project_id, sequences, wait=bool(body.get("wait", False))),
                "refit_all": lambda: facade.refit_all(project_id, wait=bool(body.get("wait", False))),
                "render_narration": lambda: facade.render_narration(project_id, wait=bool(body.get("wait", False))),
                "render_mix": lambda: facade.render_mix(project_id, wait=bool(body.get("wait", False))),
                "render_full_preview": lambda: facade.render_full_preview(project_id, wait=bool(body.get("wait", False))),
                "export_video": lambda: facade.export_video(
                    project_id, confirm_export=True, output_path=body.get("output_path"), wait=bool(body.get("wait", False))
                ),
            }
            handler = mapping.get(job_type)
            if handler is None:
                raise HTTPException(status_code=400, detail=f"Unknown job_type: {job_type}")
            return handler()
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.get(f"{prefix}/jobs")
    def http_list_jobs(
        project_id: str | None = None, status: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        return facade.list_jobs(project_id=project_id, status=status, limit=limit)

    @app.get(f"{prefix}/jobs/{{job_id}}")
    def http_get_job(job_id: str) -> dict[str, Any]:
        try:
            return facade.get_job(job_id)
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.post(f"{prefix}/jobs/{{job_id}}/cancel")
    def http_cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return facade.cancel_job(job_id)
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.get(f"{prefix}/projects/{{project_id}}/quality")
    def http_quality(project_id: str) -> dict[str, Any]:
        try:
            return facade.build_quality_report(project_id)
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.get(f"{prefix}/projects/{{project_id}}/invariants")
    def http_invariants(project_id: str) -> dict[str, Any]:
        try:
            return facade.assert_project_invariants(project_id)
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.get(f"{prefix}/projects/{{project_id}}/plan")
    def http_plan(project_id: str, force: bool = False) -> dict[str, Any]:
        try:
            return facade.get_generation_plan(project_id, force=force)
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.get(f"{prefix}/projects/{{project_id}}/snapshots")
    def http_list_snapshots(project_id: str) -> list[dict[str, Any]]:
        try:
            return facade.list_snapshots(project_id)
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.post(f"{prefix}/projects/{{project_id}}/snapshots")
    def http_create_snapshot(project_id: str, reason: str = "manual") -> dict[str, Any]:
        try:
            return facade.create_snapshot(project_id, reason=reason)
        except DubbingError as exc:
            raise _err(exc) from exc

    @app.exception_handler(DubbingError)  # type: ignore[arg-type]
    async def _dubbing_exception_handler(_request: Request, exc: DubbingError) -> JSONResponse:  # pragma: no cover
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())
