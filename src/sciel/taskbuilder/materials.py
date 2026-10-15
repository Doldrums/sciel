"""A bounded text digest of a folder of materials: documents in full where they fit, data files by their shape."""

from __future__ import annotations

import csv
import json
from pathlib import Path

TEXT = {".md", ".txt", ".rst", ".py", ".yaml", ".yml", ".toml", ".json", ".tex", ".r", ".jl", ".m", ".sql"}
MAX_FILES = 200
MAX_TEXT = 12000
MAX_TOTAL = 60000
SKIP = {".git", "__pycache__", ".sciel", "node_modules", ".venv", ".ipynb_checkpoints"}


def _notebook(path: Path) -> str:
    cells = json.loads(path.read_text()).get("cells", [])
    return "\n\n".join(f"# [{c.get('cell_type')}]\n" + "".join(c.get("source", [])) for c in cells)


def _csv(path: Path) -> str:
    with open(path, newline="", errors="replace") as stream:
        rows = []
        for i, row in enumerate(csv.reader(stream)):
            if i < 6:
                rows.append(row)
        count = i + 1 if rows else 0
    return f"{count} lines; header and first rows:\n" + "\n".join(",".join(r[:40]) for r in rows)


def _array(path: Path) -> str:
    import numpy as np
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return f"numpy array shape={array.shape} dtype={array.dtype}"


def _parquet(path: Path) -> str:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return "parquet file (install pyarrow to read its schema)"
    meta = pq.ParquetFile(path)
    return f"{meta.metadata.num_rows} rows; schema:\n{meta.schema_arrow}"


def _pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "PDF (install pypdf to read its text)"
    return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages[:30])


READERS = {".ipynb": _notebook, ".csv": _csv, ".tsv": _csv, ".npy": _array, ".parquet": _parquet, ".pdf": _pdf}


def digest(folder) -> tuple[str, list[dict]]:
    """The digest the drafting model reads, and one row per file for the screen."""
    root = Path(folder).resolve()
    if not root.is_dir():
        raise ValueError(f"{root} is not a folder")
    files = sorted(p for p in root.rglob("*") if p.is_file() and not SKIP & set(p.relative_to(root).parts))
    rows, parts, used = [], [], 0
    for path in files[:MAX_FILES]:
        suffix = path.suffix.lower()
        size = path.stat().st_size
        try:
            if suffix in READERS:
                text = READERS[suffix](path)
            elif suffix in TEXT and size <= 2_000_000:
                text = path.read_text(errors="replace")
            else:
                text = f"binary file, {size} bytes"
        except Exception as exc:
            text = f"could not read: {type(exc).__name__}: {exc}"
        text = text[:MAX_TEXT]
        room = MAX_TOTAL - used
        if room <= 0:
            text = "(not shown: the digest is full)"
        elif len(text) > room:
            text = text[:room] + "\n…"
        used += len(text)
        rel = path.relative_to(root)
        parts.append(f"=== {rel} ({size} bytes) ===\n{text}")
        rows.append({"name": str(rel), "size": size, "took": " ".join(text.split())[:90]})
    if len(files) > MAX_FILES:
        parts.append(f"... and {len(files) - MAX_FILES} more files not shown")
    return "\n\n".join(parts), rows
