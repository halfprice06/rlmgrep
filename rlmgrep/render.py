from __future__ import annotations

from .ingest import FileRecord

COLOR_RESET = "\x1b[0m"
COLOR_PATH = "\x1b[35m"
COLOR_LINE_NO = "\x1b[32m"


def _colorize(text: str, color: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{color}{text}{COLOR_RESET}"


def _format_heading(path: str, use_color: bool) -> str:
    if not path.startswith((".", "/")):
        path = f"./{path}"
    return _colorize(path, COLOR_PATH, use_color)


def _format_line(
    line_no: int,
    text: str,
    is_match: bool,
    use_color: bool,
    heading: bool,
) -> str:
    delim = ":" if is_match else "-"
    prefix = _colorize(str(line_no), COLOR_LINE_NO, use_color)
    sep = "\t" if heading else ""
    return f"{prefix}{delim}{sep}{text}"


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ranges.sort()
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def render_matches(
    files: dict[str, FileRecord],
    matches: dict[str, list[int]],
    before: int,
    after: int,
    use_color: bool = False,
    heading: bool = True,
) -> list[str]:
    output: list[str] = []

    paths = sorted(matches.keys())
    for idx, path in enumerate(paths):
        record = files.get(path)
        if record is None:
            continue
        if heading:
            if idx > 0:
                output.append("")
            output.append(_format_heading(path, use_color))
        lines = record.lines
        page_map = record.page_map
        n_lines = len(lines)
        match_lines = sorted(set(matches[path]))
        match_set = set(match_lines)

        if before == 0 and after == 0:
            for line_no in match_lines:
                if 1 <= line_no <= n_lines:
                    text = lines[line_no - 1]
                    if page_map:
                        text = f"{text}\tpage={page_map[line_no - 1]}"
                    output.append(
                        _format_line(
                            line_no,
                            text,
                            True,
                            use_color,
                            heading,
                        )
                    )
            continue

        ranges: list[tuple[int, int]] = []
        for line_no in match_lines:
            start = max(1, line_no - before)
            end = min(n_lines, line_no + after)
            ranges.append((start, end))

        merged = _merge_ranges(ranges)
        for idx, (start, end) in enumerate(merged):
            for line_no in range(start, end + 1):
                text = lines[line_no - 1]
                if page_map:
                    text = f"{text}\tpage={page_map[line_no - 1]}"
                is_match = line_no in match_set
                output.append(
                    _format_line(
                        line_no,
                        text,
                        is_match,
                        use_color,
                        heading,
                    )
                )
            if idx < len(merged) - 1:
                output.append("--")

    return output
