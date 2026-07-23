from __future__ import annotations

import pytest

from app.core.video_dubbing.duration_fitter import DurationFitter
from app.core.video_dubbing.models import (
    Alignment,
    CueStatus,
    DubbingCue,
    DubbingProjectSettings,
    FittingStrategy,
    SyncMode,
)
from app.core.video_dubbing.srt_parser import (
    SrtParseError,
    cues_from_parsed,
    cues_to_srt,
    parse_srt,
    parse_srt_file,
    parse_timestamp_ms,
    validate_cues,
)


def _make_cue(
    sequence: int,
    start_ms: int,
    end_ms: int,
    text: str = "hello",
    raw_ms: int | None = None,
) -> DubbingCue:
    return DubbingCue(
        cue_id=str(sequence),
        sequence=sequence,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_budget_ms=end_ms - start_ms,
        source_text=text,
        spoken_text=text,
        raw_duration_ms=raw_ms,
    )


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------


def test_parse_timestamp_comma_milliseconds():
    assert parse_timestamp_ms("00:00:05,000") == 5000
    assert parse_timestamp_ms("01:02:03,500") == (1 * 3600 + 2 * 60 + 3) * 1000 + 500


def test_parse_timestamp_dot_milliseconds():
    assert parse_timestamp_ms("00:00:05.250") == 5250


def test_parse_timestamp_variable_precision():
    assert parse_timestamp_ms("00:00:05,2") == 5200
    assert parse_timestamp_ms("00:00:05,25") == 5250
    assert parse_timestamp_ms("00:00:05,2500") == 5250


def test_parse_timestamp_invalid():
    with pytest.raises(SrtParseError):
        parse_timestamp_ms("not a time")


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

BASIC_SRT = """1
00:00:05,000 --> 00:00:08,000
Сегодня мы рассмотрим новую систему.

2
00:00:10,500 --> 00:00:14,000
Сначала создадим новый проект.
"""


def test_parse_basic_srt():
    cues, warnings = parse_srt(BASIC_SRT)
    assert [c.sequence for c in cues] == [1, 2]
    assert cues[0].start_ms == 5000
    assert cues[0].end_ms == 8000
    assert cues[0].text == "Сегодня мы рассмотрим новую систему."
    assert cues[1].start_ms == 10500
    assert warnings == []


def test_parse_handles_bom_and_crlf():
    srt = "\ufeff1\r\n00:00:01,000 --> 00:00:02,000\r\nLine one.\r\n"
    cues, _ = parse_srt(srt)
    assert len(cues) == 1
    assert cues[0].text == "Line one."


def test_parse_multiline_text():
    srt = (
        "1\n00:00:01,000 --> 00:00:03,000\nFirst line.\nSecond line.\n"
    )
    cues, _ = parse_srt(srt)
    assert cues[0].text == "First line.\nSecond line."


def test_parse_missing_index_assigns_implicit():
    srt = "00:00:01,000 --> 00:00:02,000\nText only.\n"
    cues, warnings = parse_srt(srt)
    assert cues[0].sequence == 1
    assert any(w.code == "missing_index" for w in warnings)


def test_parse_end_before_start_raises():
    srt = "1\n00:00:05,000 --> 00:00:05,000\nSame time.\n"
    with pytest.raises(SrtParseError):
        parse_srt(srt)


def test_parse_empty_text_raises():
    srt = "1\n00:00:01,000 --> 00:00:02,000\n\n"
    with pytest.raises(SrtParseError):
        parse_srt(srt)


def test_parse_missing_separator_raises():
    srt = "1\n00:00:01,000 - 00:00:02,000\nText.\n"
    with pytest.raises(SrtParseError):
        parse_srt(srt)


def test_parse_bad_timestamp_raises():
    srt = "1\n00:00:01,000 --> nope\nText.\n"
    with pytest.raises(SrtParseError):
        parse_srt(srt)


def test_parse_empty_content_raises():
    with pytest.raises(SrtParseError):
        parse_srt("")


def test_parse_extra_whitespace_and_blank_lines():
    srt = "\n\n1\n  00:00:01,000 --> 00:00:02,000  \n  Spaced text.  \n\n\n"
    cues, _ = parse_srt(srt)
    assert len(cues) == 1
    assert cues[0].start_ms == 1000


def test_parse_skipped_numbers_ok():
    srt = (
        "1\n00:00:01,000 --> 00:00:02,000\nA.\n"
        "5\n00:00:03,000 --> 00:00:04,000\nB.\n"
    )
    cues, _ = parse_srt(srt)
    assert [c.sequence for c in cues] == [1, 5]


def test_duplicate_index_warning():
    srt = (
        "1\n00:00:01,000 --> 00:00:02,000\nA.\n"
        "1\n00:00:03,000 --> 00:00:04,000\nB.\n"
    )
    cues, warnings = parse_srt(srt)
    assert len(cues) == 2
    assert any(w.code == "duplicate_index" for w in warnings)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_detects_overlap():
    cues, _ = parse_srt(BASIC_SRT)
    cues[1].start_ms = 7600  # overlap
    cues[1].end_ms = 9000
    warnings = validate_cues([c for c in cues], video_duration_ms=20_000)
    assert any(w.code == "overlap" for w in warnings)


def test_validate_detects_outside_video():
    cues, _ = parse_srt(BASIC_SRT)
    warnings = validate_cues(cues, video_duration_ms=12_000)
    assert any(w.code == "ends_after_video" for w in warnings)


def test_validate_detects_cue_after_video_end():
    cues, _ = parse_srt(BASIC_SRT)
    warnings = validate_cues(cues, video_duration_ms=6_000)
    assert any(w.code == "outside_video" for w in warnings)


def test_validate_short_window_warning():
    parsed, _ = parse_srt(BASIC_SRT)
    parsed[0].start_ms = 5000
    parsed[0].end_ms = 5080  # 80ms
    warnings = validate_cues(parsed, video_duration_ms=20_000)
    assert any(w.code == "very_short_window" for w in warnings)


def test_validate_clean_cues_no_warnings():
    parsed, _ = parse_srt(BASIC_SRT)
    assert validate_cues(parsed, video_duration_ms=20_000) == []


# ---------------------------------------------------------------------------
# Round-trip and file IO
# ---------------------------------------------------------------------------


def test_round_trip_srt_export(tmp_path):
    parsed, _ = parse_srt(BASIC_SRT)
    cues = cues_from_parsed(parsed)
    rendered = cues_to_srt(cues)
    reparsed, _ = parse_srt(rendered)
    assert [c.start_ms for c in reparsed] == [5000, 10500]


def test_parse_srt_file_handles_utf8_bom(tmp_path):
    srt_path = tmp_path / "sub.srt"
    srt_path.write_bytes(
        "\ufeff1\n00:00:01,000 --> 00:00:02,000\nПривет мир.\n".encode("utf-8")
    )
    parsed, _ = parse_srt_file(srt_path)
    assert parsed[0].text == "Привет мир."


def test_parse_srt_file_missing_raises(tmp_path):
    with pytest.raises(SrtParseError):
        parse_srt_file(tmp_path / "nope.srt")


# ---------------------------------------------------------------------------
# DurationFitter
# ---------------------------------------------------------------------------


def _fitter(max_factor: float = 1.35, mode: SyncMode = SyncMode.STRICT):
    settings = DubbingProjectSettings(max_speed_factor=max_factor, sync_mode=mode)
    return DurationFitter(settings)


def test_fitter_short_cue_no_speedup():
    cue = _make_cue(1, 5000, 8000, raw_ms=2600)
    result = _fitter().evaluate(cue)
    assert result.strategy == FittingStrategy.NONE
    assert result.applied_speed_factor == 1.0
    assert result.overflow_ms == 0
    assert result.status == CueStatus.RENDERED.value


def test_fitter_exact_fit_no_speedup():
    cue = _make_cue(1, 5000, 8000, raw_ms=3000)
    result = _fitter().evaluate(cue)
    assert result.strategy == FittingStrategy.NONE


def test_fitter_mild_speedup_within_limit():
    cue = _make_cue(1, 5000, 8500, raw_ms=4100)  # 4.1/3.5 = 1.171
    result = _fitter(max_factor=1.35).evaluate(cue)
    assert result.strategy == FittingStrategy.ATEMPO
    assert round(result.applied_speed_factor, 3) == 1.171
    assert result.overflow_ms == 0
    assert result.status == CueStatus.SPEED_UP.value


def test_fitter_strict_overflow_blocks_export():
    cue = _make_cue(1, 0, 2000, raw_ms=3800)  # 1.9 > 1.35
    result = _fitter(max_factor=1.35, mode=SyncMode.STRICT).evaluate(cue)
    assert result.status == CueStatus.NEEDS_SHORTENING.value
    assert result.overflow_ms > 0
    assert "needs_text_shortening" in result.warning_codes


def test_fitter_best_effort_applies_max():
    cue = _make_cue(1, 0, 2000, raw_ms=3800)
    result = _fitter(max_factor=1.35, mode=SyncMode.BEST_EFFORT).evaluate(cue)
    assert result.status == CueStatus.SPEED_UP.value
    assert result.applied_speed_factor == 1.35
    assert result.overflow_ms > 0


def test_fitter_apply_result_populates_cue():
    cue = _make_cue(1, 0, 3500, raw_ms=4100)
    fitter = _fitter()
    result = fitter.evaluate(cue)
    fitter.apply_result(cue, result)
    assert cue.applied_speed_factor == pytest.approx(1.171, rel=1e-2)
    assert cue.fitting_strategy == FittingStrategy.ATEMPO.value
    assert cue.overflow_ms == 0


def test_fitter_requires_raw_duration():
    cue = _make_cue(1, 0, 1000, raw_ms=None)
    with pytest.raises(ValueError):
        _fitter().evaluate(cue)


def test_placement_offset_alignment():
    assert DurationFitter.placement_offset(2000, 3000, Alignment.START) == 0
    assert DurationFitter.placement_offset(2000, 3000, Alignment.CENTER) == 500
    assert DurationFitter.placement_offset(2000, 3000, Alignment.END) == 1000


def test_atempo_chain_within_single_filter():
    chain = DurationFitter.build_atempo_chain(1.171)
    assert len(chain) == 1
    assert chain[0].startswith("atempo=1.171")


def test_atempo_chain_chains_for_large_factors():
    chain = DurationFitter.build_atempo_chain(4.5)
    # 4.5 -> 2.0 (rem 2.25) -> 2.0 (rem 1.125)
    assert chain.count("atempo=2.000") == 2
    assert chain[-1].startswith("atempo=1.125")


def test_atempo_chain_chains_for_small_factors():
    chain = DurationFitter.build_atempo_chain(0.3)
    # 0.3 -> 0.5 (rem 0.6)
    assert chain[0].startswith("atempo=0.5")
    assert chain[-1].startswith("atempo=0.6")


def test_fitter_very_short_window():
    cue = _make_cue(1, 0, 100, raw_ms=400)  # factor 4 > 1.35
    result = _fitter(max_factor=1.35, mode=SyncMode.STRICT).evaluate(cue)
    assert result.status == CueStatus.NEEDS_SHORTENING.value


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_project_dirs_created(tmp_path):
    from app.core.video_dubbing.models import DubbingProject

    project = DubbingProject(
        project_id="abc",
        project_dir=tmp_path / "proj",
    )
    project.ensure_directories()
    assert project.cues_dir().is_dir()
    assert project.render_dir().is_dir()
    assert project.temp_dir().is_dir()


def test_settings_round_trip():
    from app.core.video_dubbing.models import (
        DubbingProjectSettings,
        DuckingSettings,
        OriginalAudioMode,
    )

    settings = DubbingProjectSettings(
        language="ru",
        max_speed_factor=1.4,
        ducking=DuckingSettings(mode=OriginalAudioMode.CONSTANT),
    )
    data = settings.to_dict()
    restored = DubbingProjectSettings.from_dict(data)
    assert restored.language == "ru"
    assert restored.max_speed_factor == 1.4
    assert restored.ducking.mode == OriginalAudioMode.CONSTANT


def test_cue_mark_stale():
    cue = _make_cue(1, 0, 1000)
    cue.status = CueStatus.FITTED.value
    cue.mark_stale()
    assert cue.is_stale is True
    assert cue.status == CueStatus.STALE.value
