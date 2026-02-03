from __future__ import annotations

from .ingest import FileRecord


def _format_line(
    path: str,
    line_no: int,
    text: str,
    is_match: bool,
    show_filename: bool,
    show_line_numbers: bool,
) -> str:
    delim = ":" if is_match else "-"
    if show_filename and show_line_numbers:
        return f"{path}{delim}{line_no}{delim}{text}"
    if show_filename:
        return f"{path}{delim}{text}"
    if show_line_numbers:
        return f"{line_no}{delim}{text}"
    return text


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
    show_line_numbers: bool,
    show_filename: bool,
    before: int,
    after: int,
) -> list[str]:
    output: list[str] = []
    multiple_files = len(files) > 1
    show_filename = show_filename or multiple_files

    for path in sorted(matches.keys()):
        record = files.get(path)
        if record is None:
            continue
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
                            path,
                            line_no,
                            text,
                            True,
                            show_filename,
                            show_line_numbers,
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
                        path,
                        line_no,
                        text,
                        is_match,
                        show_filename,
                        show_line_numbers,
                    )
                )
            if idx < len(merged) - 1:
                output.append("--")

    return output
