# Video Dubbing MCP — Manual Smoke Test & Agent Reference

This document describes how to drive the **video-dubbing** pipeline through the
existing persistent engine host (`engine_host.py`) using MCP tools or the
HTTP API. No second MCP server is created; everything is registered on the same
`/mcp` endpoint and `/api/video-dubbing/*` routes as the audiobook tools.

## 1. Start the engine host

```bash
python engine_host.py --host 127.0.0.1 --port 8765
```

Check health:

```bash
curl http://127.0.0.1:8765/health
# {"status":"ok","name":"LocalText2Voice","mcp_endpoint":"http://127.0.0.1:8765/mcp"}
```

If `local_server.auth_token` is set, pass `Authorization: Bearer <token>` (or
`?token=<token>`) on every request. The same token guards `/api/video-dubbing/*`.

## 2. Connect an MCP client

Point Claude Desktop / OpenCode / any MCP client at:

```
http://127.0.0.1:8765/mcp
```

Read the resources first:

```
localtext2voice://video-dubbing/docs
localtext2voice://video-dubbing/actions
localtext2voice://video-dubbing/settings-schema
localtext2voice://video-dubbing/quality-model
localtext2voice://video-dubbing/scenario-schema
```

## 3. End-to-end workflow

```text
dubbing_create_project(project_dir=..., title="Demo")
  -> dubbing_attach_video(project_id, video_path=...)
  -> dubbing_import_srt(project_id, srt_path=...)
  -> dubbing_analyze_project(project_id)
  -> dubbing_list_cues(project_id, has_overflow=true)            # find problems
  -> dubbing_simulate_timing_changes(project_id, [37,38,39], {...})   # dry-run
  -> dubbing_compare_variants(...)                               # pick lowest score
  -> dubbing_optimize_timing(..., apply=true, expected_revision=N)
  -> dubbing_get_generation_plan(project_id)                     # reuse categories
  -> dubbing_generate_cues(project_id, [37,38], force=false)     # job, returns job_id
  -> dubbing_refit_cues(project_id, [37,38,39])                  # re-fit, no TTS
  -> dubbing_render_full_preview(project_id)
  -> dubbing_render_narration(project_id) -> dubbing_render_mix(project_id)
  -> dubbing_export_video(project_id, confirm_export=true)
  -> dubbing_validate_final_video(project_id)
  -> dubbing_build_quality_report(project_id)
  -> dubbing_collect_diagnostic_bundle(project_id)
```

### HTTP equivalents

```bash
# Create project
curl -X POST http://127.0.0.1:8765/api/video-dubbing/projects \
  -H 'Authorization: Bearer TOKEN' -H 'Content-Type: application/json' \
  -d '{"project_dir":"C:/dub/demo","title":"Demo"}'

# Attach video
curl -X POST http://127.0.0.1:8765/api/video-dubbing/projects/<id>/video \
  -H 'Authorization: Bearer TOKEN' -H 'Content-Type: application/json' \
  -d '{"video_path":"C:/source/movie.mp4"}'

# Import SRT
curl -X POST http://127.0.0.1:8765/api/video-dubbing/projects/<id>/srt \
  -H 'Authorization: Bearer TOKEN' -H 'Content-Type: application/json' \
  -d '{"srt_path":"C:/source/subs.srt"}'

# Simulate
curl -X POST http://127.0.0.1:8765/api/video-dubbing/projects/<id>/simulate \
  -H 'Authorization: Bearer TOKEN' -H 'Content-Type: application/json' \
  -d '{"sequences":[37,38,39],"changes":{"preferred_speed_limit":1.25,"max_shift_ms":900}}'

# Submit a generate job
curl -X POST http://127.0.0.1:8765/api/video-dubbing/projects/<id>/jobs \
  -H 'Authorization: Bearer TOKEN' -H 'Content-Type: application/json' \
  -d '{"job_type":"generate_missing"}'

# Poll + cancel
curl http://127.0.0.1:8765/api/video-dubbing/jobs/<job_id>
curl -X POST http://127.0.0.1:8765/api/video-dubbing/jobs/<job_id>/cancel
```

## 4. Concurrency model

Every mutation accepts `expected_revision`. If it does not match the current
revision you receive:

```json
{"error":"project_changed","expected_revision":84,"current_revision":85}
```

Re-read `dubbing_get_project_state`, update `expected_revision`, and retry. Only
one mutating job may run per project; others get `error=project_busy`.

## 5. Jobs & cancellation

Heavy operations are asynchronous. They return a `job_id`. Poll with
`dubbing_get_job` / `GET /api/video-dubbing/jobs/{job_id}` and cancel with
`dubbing_cancel_job` / `POST .../cancel`. Cancellation is cooperative: the
active FFmpeg/TTS subprocess is stopped, leftover `.part` files are moved to
`<project>/temp/recovery`, no cue is left in `rendering`, and the previous valid
final artifact is preserved.

## 6. Quality model

`dubbing_compare_variants` and `dubbing_optimize_timing` rank candidates with a
deterministic cost function (lower is better). The breakdown explains the
contribution of overflow, overlap, speed excess, shift, tempo jumps, guard-gap
violations, boundary violations, missing artifacts and regeneration cost.
Weights are configurable via the `QualityWeights` dataclass.

## 7. Security

* Only allowlisted domain actions are accepted (`dubbing_execute_action`).
* Arbitrary `execute_python` / `execute_shell` / `execute_sql` /
  `run_arbitrary_ffmpeg` / `eval` / `exec` are rejected.
* Paths are normalised; path traversal (`..`) is forbidden.
* Auth token redaction is applied; full cue text is never written to the
  observability JSONL (only `text_sha256` and `text_length`).
* Fault injection is gated behind `diagnostic_test_mode`, localhost-only, a
  separate `diagnostic_test_token`, and `test_project=true`.

## 8. Scenario runner (resilience tests)

```json
{
  "name": "cancel_during_mux",
  "project_id": "<id>",
  "steps": [
    {"action": "generate_missing"},
    {"action": "render_narration"},
    {"action": "render_mix"},
    {"action": "export_video", "arguments": {"confirm_export": true}},
    {"assert": {"job_status": "completed"}}
  ]
}
```

Call via `dubbing_run_scenario`. The runner uses the same facade + job manager
as MCP/HTTP — never a separate test implementation.
