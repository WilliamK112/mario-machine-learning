from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mario Training Viewer</title>
  <style>
    :root { color-scheme: dark; font-family: Segoe UI, Arial, sans-serif; }
    body { margin: 0; background: #101418; color: #f4f7fb; }
    main { min-height: 100vh; display: grid; grid-template-columns: minmax(0, 1fr) 320px; }
    .stage { display: grid; place-items: center; background: #07090c; padding: 20px; }
    video { width: min(100%, 980px); max-height: 92vh; image-rendering: pixelated; background: #000; }
    aside { border-left: 1px solid #29313b; padding: 18px; background: #151b22; }
    h1 { font-size: 18px; margin: 0 0 14px; }
    h2 { font-size: 13px; margin: 18px 0 8px; color: #b7c4d4; text-transform: uppercase; }
    dl { display: grid; grid-template-columns: 130px 1fr; gap: 8px 12px; margin: 0; font-size: 13px; }
    dt { color: #91a1b3; }
    dd { margin: 0; word-break: break-word; }
    .pill { display: inline-block; padding: 3px 8px; border: 1px solid #3a4654; border-radius: 999px; }
    .good { color: #8ff0a4; }
    .warn { color: #ffd479; }
    button { margin-top: 16px; width: 100%; padding: 9px 12px; border: 1px solid #445166; background: #202937; color: #fff; border-radius: 6px; cursor: pointer; }
    button:hover { background: #293449; }
    @media (max-width: 860px) { main { grid-template-columns: 1fr; } aside { border-left: 0; border-top: 1px solid #29313b; } }
  </style>
</head>
<body>
  <main>
    <section class="stage">
      <video id="video" controls autoplay muted loop playsinline></video>
    </section>
    <aside>
      <h1>Mario Training Viewer</h1>
      <dl>
        <dt>Video</dt><dd id="videoName">-</dd>
        <dt>Updated</dt><dd id="updated">-</dd>
        <dt>Step</dt><dd id="step">-</dd>
        <dt>Furthest</dt><dd id="furthest">-</dd>
        <dt>Stage clears</dt><dd id="clears">-</dd>
        <dt>Best max_x</dt><dd id="maxX">-</dd>
      </dl>
      <h2>Best eval</h2>
      <dl>
        <dt>Step</dt><dd id="bestStep">-</dd>
        <dt>Furthest</dt><dd id="bestFurthest">-</dd>
        <dt>Clears</dt><dd id="bestClears">-</dd>
      </dl>
      <h2>State</h2>
      <p id="state" class="pill">Loading</p>
      <button id="reload">Reload video</button>
    </aside>
  </main>
  <script>
    const video = document.getElementById("video");
    let currentKey = "";

    function setText(id, value) {
      document.getElementById(id).textContent = value ?? "-";
    }

    async function refresh(force = false) {
      const res = await fetch("/api/status?t=" + Date.now(), { cache: "no-store" });
      const status = await res.json();
      setText("videoName", status.video_name);
      setText("updated", status.video_mtime);
      setText("step", status.preview_step ?? status.eval_step);
      setText("furthest", status.eval_best_world_stage);
      setText("clears", status.eval_stage_clears);
      setText("maxX", status.eval_best_max_x);
      setText("bestStep", status.best_step);
      setText("bestFurthest", status.best_world_stage);
      setText("bestClears", status.best_stage_clears);
      const state = document.getElementById("state");
      state.textContent = status.message;
      state.className = "pill " + (status.has_video ? "good" : "warn");
      if (status.has_video && (force || status.video_key !== currentKey)) {
        currentKey = status.video_key;
        video.src = "/video/latest.mp4?key=" + encodeURIComponent(currentKey);
        video.load();
        video.play().catch(() => {});
      }
    }

    document.getElementById("reload").addEventListener("click", () => refresh(true));
    refresh(true);
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


def numeric_step(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.as_posix())
    return int(match.group(1)) if match else -1


def latest_file(paths: list[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: (numeric_step(p), p.stat().st_mtime))


def load_json(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_latest_preview(run_dir: Path) -> Path | None:
    return latest_file(list((run_dir / "previews").glob("step_*/preview.mp4")))


def find_latest_eval_summary(run_dir: Path) -> Path | None:
    return latest_file(list((run_dir / "evaluations").glob("step_*/summary.json")))


def as_world_stage(value) -> str:
    if isinstance(value, list | tuple) and len(value) >= 2:
        return f"{value[0]}-{value[1]}"
    return "-"


class ViewerHandler(BaseHTTPRequestHandler):
    run_dir: Path

    def send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/status":
            self.send_status()
            return
        if parsed.path == "/video/latest.mp4":
            self.send_latest_video()
            return
        self.send_error(int(HTTPStatus.NOT_FOUND), "not found")

    def send_status(self) -> None:
        preview = find_latest_preview(self.run_dir)
        eval_summary_path = find_latest_eval_summary(self.run_dir)
        eval_summary = load_json(eval_summary_path)
        best_summary = load_json(self.run_dir / "evaluations" / "best_eval_summary.json")
        best_inner = best_summary.get("summary") or {}

        payload = {
            "run_dir": str(self.run_dir),
            "has_video": preview is not None,
            "video_name": str(preview.relative_to(self.run_dir)) if preview else None,
            "video_key": f"{numeric_step(preview)}-{preview.stat().st_mtime_ns}" if preview else "",
            "video_mtime": preview.stat().st_mtime if preview else None,
            "preview_step": numeric_step(preview) if preview else None,
            "eval_step": eval_summary.get("num_timesteps"),
            "eval_best_world_stage": as_world_stage(eval_summary.get("best_world_stage")),
            "eval_stage_clears": (
                f"{eval_summary.get('best_stage_clears', '-')}/{eval_summary.get('average_stage_clears', '-')}"
            ),
            "eval_best_max_x": eval_summary.get("best_max_x"),
            "best_step": best_summary.get("num_timesteps"),
            "best_world_stage": as_world_stage(best_inner.get("best_world_stage")),
            "best_stage_clears": (
                f"{best_inner.get('best_stage_clears', '-')}/{best_inner.get('average_stage_clears', '-')}"
            ),
            "message": "Watching latest preview" if preview else "No preview video yet",
        }
        self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json")

    def send_latest_video(self) -> None:
        preview = find_latest_preview(self.run_dir)
        if preview is None:
            self.send_error(int(HTTPStatus.NOT_FOUND), "no preview video yet")
            return
        data = preview.read_bytes()
        content_type = mimetypes.guess_type(preview.name)[0] or "video/mp4"
        self.send_bytes(data, content_type)

    def log_message(self, format: str, *args) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a small auto-refreshing Mario training viewer.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    ViewerHandler.run_dir = run_dir
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(f"viewer_ok url=http://{args.host}:{args.port}/ run_dir={run_dir}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
