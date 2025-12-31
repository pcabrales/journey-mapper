import os
import uuid
import pathlib
import subprocess

from flask import Flask, abort, redirect, render_template, request, send_file, url_for

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SCRIPT = BASE_DIR / "journey_mapper.py"

app = Flask(__name__)

# Prevent overly large uploads (adjust as desired).
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MiB

OUTPUT_ROOT = pathlib.Path("/tmp/journey_outputs")


def run_mapper(csv_path: pathlib.Path, title: str, want_video: bool) -> str:
    run_id = uuid.uuid4().hex
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    out_html = out_dir / "journey.html"
    out_video = out_dir / "journey.mp4"

    cmd = [
        "python",
        str(SCRIPT),
        str(csv_path),
        "--output",
        str(out_html),
        "--title",
        title,
    ]

    if want_video:
        cmd += ["--video", str(out_video)]

    env = os.environ.copy()

    subprocess.run(cmd, check=True, cwd=str(BASE_DIR), env=env)

    return run_id


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/generate")
def generate():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    title = (request.form.get("title") or "Journey Mapper").strip()
    want_video = request.form.get("want_video") == "on"

    mode = request.form.get("mode") or "example"

    if mode == "example":
        csv_path = DATA_DIR / "example_journey.csv"
        if not csv_path.exists():
            abort(500, "Example CSV not found in data/example_journey.csv")
    else:
        upload = request.files.get("csv")
        if upload is None or upload.filename == "":
            abort(400, "Missing CSV file upload.")

        tmp_csv = OUTPUT_ROOT / f"upload_{uuid.uuid4().hex}.csv"
        upload.save(tmp_csv)
        csv_path = tmp_csv

    run_id = run_mapper(csv_path, title, want_video)
    return redirect(url_for("result", run_id=run_id))


@app.get("/result/<run_id>")
def result(run_id: str):
    out_dir = OUTPUT_ROOT / run_id
    if not out_dir.exists():
        abort(404)

    has_video = (out_dir / "journey.mp4").exists()
    return render_template("result.html", run_id=run_id, has_video=has_video)


@app.get("/map/<run_id>")
def map_html(run_id: str):
    out_html = OUTPUT_ROOT / run_id / "journey.html"
    if not out_html.exists():
        abort(404)
    return send_file(out_html, mimetype="text/html")


@app.get("/video/<run_id>")
def video(run_id: str):
    out_video = OUTPUT_ROOT / run_id / "journey.mp4"
    if not out_video.exists():
        abort(404)
    return send_file(out_video, mimetype="video/mp4")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
