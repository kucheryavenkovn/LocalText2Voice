from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .models import DubbingCue


class SrtParseError(ValueError):
    pass


@dataclass
class SrtWarning:
    code: str
    message: str
    sequence: int | None = None


_TIME_SEPARATOR = re.compile(r"-->")
_TIMESTAMP = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{1,2}):(?P<s>\d{1,2})(?<![.,])(?P<sep>[.,])(?P<ms>\d{1,6})"
)


def parse_timestamp_ms(value: str) -> int:
    match = _TIMESTAMP.search(value.strip())
    if match is None:
        raise SrtParseError(f"Invalid timestamp: {value!r}")
    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    ms_raw = match.group("ms")
    ms_digits = len(ms_raw)
    milliseconds = int(ms_raw)
    if ms_digits == 1:
        milliseconds *= 100
    elif ms_digits == 2:
        milliseconds *= 10
    elif ms_digits > 3:
        milliseconds //= 10 ** (ms_digits - 3)
    total_ms = (
        hours * 3_600_000
        + minutes * 60_000
        + seconds * 1_000
        + milliseconds
    )
    return total_ms


def _format_ms(ms: int) -> str:
    if ms < 0:
        ms = 0
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, milliseconds = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


@dataclass
class ParsedCue:
    sequence: int
    start_ms: int
    end_ms: int
    text: str


def parse_srt(content: str) -> tuple[list[ParsedCue], list[SrtWarning]]:
    text = _strip_bom(content)
    normalized_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    warnings: list[SrtWarning] = []
    cues: list[ParsedCue] = []

    blocks = _split_blocks(normalized_lines)
    seen_sequences: set[int] = set()
    implicit_sequence = 0
    for index, block in enumerate(blocks, start=1):
        parsed_block = _parse_block(block, index, warnings)
        if parsed_block is None:
            continue
        sequence, start_ms, end_ms, cue_text = parsed_block
        if sequence is None:
            implicit_sequence += 1
            sequence = implicit_sequence
            warnings.append(
                SrtWarning(
                    code="missing_index",
                    message=f"Block {index} has no index number; assigned #{sequence}.",
                    sequence=sequence,
                )
            )
        else:
            implicit_sequence = max(implicit_sequence, sequence)
        if sequence in seen_sequences:
            warnings.append(
                SrtWarning(
                    code="duplicate_index",
                    message=f"Duplicate cue index #{sequence}; kept as-is.",
                    sequence=sequence,
                )
            )
        seen_sequences.add(sequence)
        if end_ms <= start_ms:
            raise SrtParseError(
                f"Cue #{sequence}: end time must be after start time "
                f"({_format_ms(start_ms)} -> {_format_ms(end_ms)})."
            )
        cues.append(
            ParsedCue(
                sequence=sequence,
                start_ms=start_ms,
                end_ms=end_ms,
                text=cue_text,
            )
        )

    if not cues:
        raise SrtParseError("No valid subtitle cues found in SRT content.")
    cues.sort(key=lambda cue: (cue.start_ms, cue.sequence))
    return cues, warnings


def parse_srt_file(path: Path) -> tuple[list[ParsedCue], list[SrtWarning]]:
    if not path.is_file():
        raise SrtParseError(f"SRT file not found: {path}")
    for encoding in ("utf-8-sig", "utf-8", "cp1251", "cp1252"):
        try:
            content = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            raise SrtParseError(f"Could not read {path.name}: {exc}") from exc
    else:
        raise SrtParseError(
            f"Could not decode {path.name}. Save it as UTF-8 and try again."
        )
    return parse_srt(content)


def _strip_bom(content: str) -> str:
    if content and content[0] == "\ufeff":
        return content[1:]
    return content


def _split_blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    has_timing = False
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        is_index_candidate = bool(re.fullmatch(r"\d{1,6}", stripped))
        starts_new_cue = (
            bool(current)
            and has_timing
            and is_index_candidate
            and index + 1 < len(lines)
            and _TIME_SEPARATOR.search(lines[index + 1].strip()) is not None
        )
        if starts_new_cue:
            blocks.append(_trim_block(current))
            current = [raw_line]
            has_timing = False
        elif not stripped and current and has_timing:
            blocks.append(_trim_block(current))
            current = []
            has_timing = False
        else:
            if _TIME_SEPARATOR.search(stripped) is not None:
                has_timing = True
            current.append(raw_line)
    if current:
        trimmed = _trim_block(current)
        if trimmed:
            blocks.append(trimmed)
    return [block for block in blocks if block]


def _trim_block(block: list[str]) -> list[str]:
    start = 0
    end = len(block)
    while start < end and not block[start].strip():
        start += 1
    while end > start and not block[end - 1].strip():
        end -= 1
    return block[start:end]


def _parse_block(
    block: list[str],
    block_index: int,
    warnings: list[SrtWarning],
) -> tuple[int | None, int, int, str] | None:
    if not block:
        return None
    cursor = 0
    first_line = block[0].strip()
    sequence: int | None = None
    if re.fullmatch(r"\d{1,6}", first_line):
        try:
            sequence = int(first_line)
        except ValueError:
            sequence = None
        cursor = 1
    if cursor >= len(block):
        raise SrtParseError(
            f"Block {block_index}: missing timecode line."
        )
    timing_line = block[cursor].strip()
    cursor += 1
    if _TIME_SEPARATOR.search(timing_line) is None:
        raise SrtParseError(
            f"Block {block_index}: missing '-->' separator in timecode line."
        )
    try:
        start_ms, end_ms = _parse_timing_line(timing_line)
    except SrtParseError as exc:
        raise SrtParseError(f"Block {block_index}: {exc}") from exc
    text_lines = [line.rstrip() for line in block[cursor:]]
    text = "\n".join(text_lines).strip()
    if not text:
        raise SrtParseError(f"Block {block_index}: empty cue text.")
    return sequence, start_ms, end_ms, text


def _parse_timing_line(line: str) -> tuple[int, int]:
    parts = _TIME_SEPARATOR.split(line, maxsplit=1)
    if len(parts) != 2:
        raise SrtParseError(f"Invalid timecode line: {line!r}")
    start_ms = parse_timestamp_ms(parts[0])
    end_ms = parse_timestamp_ms(parts[1])
    return start_ms, end_ms


def validate_cues(
    cues: list[ParsedCue],
    video_duration_ms: int | None = None,
) -> list[SrtWarning]:
    warnings: list[SrtWarning] = []
    minimum_window_ms = 120
    for cue in cues:
        duration_ms = cue.end_ms - cue.start_ms
        if duration_ms < minimum_window_ms:
            warnings.append(
                SrtWarning(
                    code="very_short_window",
                    message=(
                        f"Cue #{cue.sequence} window is very short "
                        f"({duration_ms} ms)."
                    ),
                    sequence=cue.sequence,
                )
            )
        if video_duration_ms and video_duration_ms > 0:
            if cue.start_ms >= video_duration_ms:
                warnings.append(
                    SrtWarning(
                        code="outside_video",
                        message=(
                            f"Cue #{cue.sequence} starts after the video ends "
                            f"at {_format_ms(cue.start_ms)}."
                        ),
                        sequence=cue.sequence,
                    )
                )
            elif cue.end_ms > video_duration_ms:
                warnings.append(
                    SrtWarning(
                        code="ends_after_video",
                        message=(
                            f"Cue #{cue.sequence} ends after the video "
                            f"({_format_ms(cue.end_ms)} > "
                            f"{_format_ms(video_duration_ms)})."
                        ),
                        sequence=cue.sequence,
                    )
                )
    sorted_cues = sorted(cues, key=lambda item: (item.start_ms, item.sequence))
    for previous, current in zip(sorted_cues, sorted_cues[1:]):
        if current.start_ms < previous.end_ms:
            warnings.append(
                SrtWarning(
                    code="overlap",
                    message=(
                        f"Cue #{current.sequence} overlaps cue #{previous.sequence} "
                        f"({_format_ms(current.start_ms)} < "
                        f"{_format_ms(previous.end_ms)})."
                    ),
                    sequence=current.sequence,
                )
            )
    return warnings


def cues_from_parsed(
    parsed: list[ParsedCue],
    base_dir: Path | None = None,
) -> list[DubbingCue]:
    _ = base_dir
    result: list[DubbingCue] = []
    for item in parsed:
        result.append(
            DubbingCue(
                cue_id=str(item.sequence),
                sequence=item.sequence,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
                duration_budget_ms=item.end_ms - item.start_ms,
                source_text=item.text,
                spoken_text=item.text,
            )
        )
    return result


def cues_to_srt(cues: list[DubbingCue]) -> str:
    lines: list[str] = []
    for cue in sorted(cues, key=lambda item: (item.start_ms, item.sequence)):
        lines.append(str(cue.sequence))
        lines.append(f"{_format_ms(cue.start_ms)} --> {_format_ms(cue.end_ms)}")
        lines.append(cue.spoken_text.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"
