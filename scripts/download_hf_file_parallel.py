#!/usr/bin/env python3
"""Download a large Hugging Face file with resumable HTTP range requests."""

from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--chunk-mb", type=int, default=32)
    parser.add_argument(
        "--request-mb",
        type=int,
        default=1,
        help="maximum bytes per HTTP range request in MiB (default: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="concurrent HTTP range requests (default: 2 to avoid overloading local proxies)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=30,
        help="maximum attempts per range; useful with flaky proxy connections",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.size <= 0
        or args.chunk_mb <= 0
        or args.request_mb <= 0
        or args.workers <= 0
        or args.retries <= 0
    ):
        raise SystemExit("size, chunk-mb, request-mb, workers, and retries must be positive")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size == args.size:
        print(f"Already complete: {output} ({args.size} bytes)", flush=True)
        return
    # Keep different chunk-size runs isolated so an interrupted retry cannot
    # confuse the progress/accounting of another run.
    part_dir = output.parent / f".{output.name}.parts-{args.chunk_mb}mb"
    part_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = args.chunk_mb * 1024 * 1024
    request_size = args.request_mb * 1024 * 1024
    ranges = [
        (start, min(start + chunk_size, args.size) - 1)
        for start in range(0, args.size, chunk_size)
    ]
    completed = 0
    progress_lock = threading.Lock()

    def download_range(item: tuple[int, int]) -> Path:
        start, end = item
        part = part_dir / f"part-{start:012d}-{end:012d}"
        expected = end - start + 1
        if part.exists() and part.stat().st_size == expected:
            return part

        temporary = part.with_suffix(".tmp")
        failures = 0
        while failures < args.retries:
            try:
                received = temporary.stat().st_size if temporary.exists() else 0
                if received > expected:
                    temporary.unlink()
                    received = 0
                if received == expected:
                    os.replace(temporary, part)
                    return part
                request_start = start + received
                request_end = min(request_start + request_size - 1, end)
                with requests.get(
                    args.url,
                    headers={"Range": f"bytes={request_start}-{request_end}"},
                    stream=True,
                    # A range is at most a few MiB.  On a flaky local proxy it
                    # is faster to retry a stalled request than to pin a worker
                    # for several minutes.
                    timeout=(15, 10),
                    allow_redirects=True,
                ) as response:
                    if response.status_code != 206:
                        raise RuntimeError(f"HTTP {response.status_code}")
                    content_range = response.headers.get("content-range", "")
                    if not content_range.startswith(f"bytes {request_start}-{request_end}/"):
                        raise RuntimeError(f"unexpected Content-Range: {content_range!r}")
                    with temporary.open("ab") as handle:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if block:
                                handle.write(block)
                if temporary.stat().st_size == expected:
                    os.replace(temporary, part)
                    return part
            except Exception as exc:
                failures += 1
                if failures == args.retries:
                    raise RuntimeError(f"range {start}-{end} failed: {exc}") from exc
                time.sleep(min(2, 2 ** min(failures - 1, 2)))

        raise AssertionError("unreachable")

    print(
        f"Downloading {args.size / 1024**3:.2f} GiB in {len(ranges)} ranges "
        f"with {args.workers} workers.",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_range, item) for item in ranges]
        for future in as_completed(futures):
            future.result()
            with progress_lock:
                completed += 1
                if completed == 1 or completed % 8 == 0 or completed == len(ranges):
                    print(f"ranges complete: {completed}/{len(ranges)}", flush=True)

    assembled = output.with_suffix(output.suffix + ".partial")
    with assembled.open("wb") as destination:
        for start, end in ranges:
            part = part_dir / f"part-{start:012d}-{end:012d}"
            with part.open("rb") as source:
                while block := source.read(16 * 1024 * 1024):
                    destination.write(block)
    if assembled.stat().st_size != args.size:
        raise RuntimeError(f"assembled file has unexpected size: {assembled.stat().st_size}")
    os.replace(assembled, output)
    print(f"Saved {output} ({output.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
