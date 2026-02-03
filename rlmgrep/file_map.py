from __future__ import annotations

def build_file_map(paths: list[str]) -> str:
    """Build an ASCII tree for normalized paths (using / separators)."""
    tree: dict[str, dict] = {}
    for path in sorted(paths):
        parts = [p for p in path.split("/") if p]
        node = tree
        for part in parts:
            node = node.setdefault(part, {})

    lines: list[str] = []

    def render(node: dict[str, dict], prefix: str = "") -> None:
        entries = list(node.items())
        for idx, (name, child) in enumerate(entries):
            is_last = idx == len(entries) - 1
            connector = "`-- " if is_last else "|-- "
            lines.append(prefix + connector + name)
            if child:
                next_prefix = prefix + ("    " if is_last else "|   ")
                render(child, next_prefix)

    render(tree)
    return "\n".join(lines)
