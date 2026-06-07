from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a YouTube reference video with yt-dlp without using browser cookies "
            "unless an explicit cookies file is provided."
        )
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cookies-file", default="", help="Optional explicit cookies.txt path.")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument("--merge-format", default="mp4")
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yt_dlp = shutil.which("yt-dlp")
    if yt_dlp is None:
        summary = {
            "ok": False,
            "error": "yt-dlp was not found on PATH",
            "url": args.url,
            "time": datetime.now().isoformat(timespec="seconds"),
        }
        (output_dir / "fetch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 1

    output_template = str(output_dir / "%(id)s.%(ext)s")
    format_selector = f"bv*[height<={int(args.max_height)}]+ba/b[height<={int(args.max_height)}]/b"
    cmd = [
        yt_dlp,
        "--no-playlist",
        "--write-info-json",
        "--merge-output-format",
        args.merge_format,
        "--output",
        output_template,
        "--format",
        format_selector,
    ]
    if args.metadata_only:
        cmd.append("--skip-download")
    if args.cookies_file:
        cmd += ["--cookies", str(Path(args.cookies_file))]
    cmd.append(args.url)

    proc = subprocess.run(cmd, text=True, cwd=output_dir, capture_output=True, check=False)
    videos = sorted(
        [
            path
            for path in output_dir.iterdir()
            if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    info_json = sorted(output_dir.glob("*.info.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    summary = {
        "ok": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "url": args.url,
        "output_dir": output_dir.resolve(),
        "video": videos[0].resolve() if videos else "",
        "info_json": info_json[0].resolve() if info_json else "",
        "used_cookies_file": bool(args.cookies_file),
        "metadata_only": bool(args.metadata_only),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "fetch_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(summary), indent=2))
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
