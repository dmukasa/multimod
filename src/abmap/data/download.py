"""Download helpers for the ABMAP supplementary datasets."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import requests
from tqdm import tqdm


SUPPLEMENTARY_FILES = (
    {
        "name": "supplementary_dataset_s1",
        "url": "https://www.pnas.org/doi/suppl/10.1073/pnas.2418918121/suppl_file/pnas.2418918121.sd01.xlsx",
        "filename": "pnas.2418918121.sd01.xlsx",
        "description": "Primary binding measurements described as Supplementary Dataset S1.",
    },
    {
        "name": "supplementary_dataset_s2",
        "url": "https://www.pnas.org/doi/suppl/10.1073/pnas.2418918121/suppl_file/pnas.2418918121.sd02.xlsx",
        "filename": "pnas.2418918121.sd02.xlsx",
        "description": "Antigen metadata referenced in Supplementary Dataset S2.",
    },
)


@dataclass
class DownloadResult:
    path: Path
    url: str
    sha256: str


def _compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, *, chunk_size: int = 1 << 20) -> DownloadResult:
    """Download a single file into ``destination`` and report its checksum."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        progress = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {destination.name}",
        )
        with open(destination, "wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                handle.write(chunk)
                progress.update(len(chunk))
        progress.close()

    checksum = _compute_sha256(destination)
    return DownloadResult(path=destination, url=url, sha256=checksum)


def download_supplementary(
    root: Path, *, selection: Optional[Iterable[str]] = None
) -> list[tuple[dict, DownloadResult]]:
    """Download all (or a subset of) supplementary files into ``root``."""

    allowed = None if selection is None else {name.lower() for name in selection}
    results = []
    for entry in SUPPLEMENTARY_FILES:
        name = entry["name"]
        if allowed is not None and name.lower() not in allowed:
            continue
        destination = root / entry["filename"]
        result = download_file(entry["url"], destination)
        results.append((entry, result))
    return results


def main(argv: Optional[list[str]] = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Download the ABMAP supplementary datasets.")
    parser.add_argument("output", type=Path, help="Directory where the supplementary files will be stored.")
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional subset of supplementary file names to download (case-insensitive).",
    )
    args = parser.parse_args(argv)

    results = download_supplementary(args.output, selection=args.only)
    payload = [
        {
            "name": entry["name"],
            "url": entry["url"],
            "path": str(result.path),
            "sha256": result.sha256,
            "description": entry["description"],
        }
        for entry, result in results
    ]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
