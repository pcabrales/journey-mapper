import csv
import os
import uuid
import pathlib
import subprocess

from flask import Flask, abort, redirect, render_template, request, send_file, url_for

BASE_DIR = pathlib.Path(__file__).resolve().parent
SCRIPT = BASE_DIR / "journey_mapper.py"

app = Flask(__name__)

# Prevent overly large uploads (adjust as desired).
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MiB

OUTPUT_ROOT = pathlib.Path("/tmp/journey_outputs")
DEFAULT_ROWS = [
    {
        "city": "Lisbon",
        "country": "Portugal",
        "latitude": 38.7223,
        "longitude": -9.1393,
        "date": "",
        "notes": "",
    },
    {
        "city": "Bangkok",
        "country": "Thailand",
        "latitude": 13.7563,
        "longitude": 100.5018,
        "date": "",
        "notes": "",
    },
]


def build_rows_from_form(form):
    cities = [value.strip() for value in form.getlist("city")]
    countries = [value.strip() for value in form.getlist("country")]
    latitudes = [value.strip() for value in form.getlist("latitude")]
    longitudes = [value.strip() for value in form.getlist("longitude")]
    dates = [value.strip() for value in form.getlist("date")]
    notes = [value.strip() for value in form.getlist("notes")]
    field_values = (cities, countries, latitudes, longitudes, dates, notes)

    if not any(value for values in field_values for value in values):
        return [row.copy() for row in DEFAULT_ROWS]

    row_count = max(
        len(cities),
        len(countries),
        len(latitudes),
        len(longitudes),
        len(dates),
        len(notes),
        0,
    )

    rows = []
    for idx in range(row_count):
        city = cities[idx] if idx < len(cities) else ""
        country = countries[idx] if idx < len(countries) else ""
        latitude = latitudes[idx] if idx < len(latitudes) else ""
        longitude = longitudes[idx] if idx < len(longitudes) else ""
        date = dates[idx] if idx < len(dates) else ""
        note = notes[idx] if idx < len(notes) else ""

        if not any([city, country, latitude, longitude, date, note]):
            continue

        if not all([city, country, latitude, longitude]):
            abort(400, f"Row {idx + 1} is missing a required field.")

        try:
            latitude_value = float(latitude)
            longitude_value = float(longitude)
        except ValueError:
            abort(400, f"Row {idx + 1} has invalid latitude/longitude.")

        rows.append(
            {
                "city": city,
                "country": country,
                "latitude": latitude_value,
                "longitude": longitude_value,
                "date": date,
                "notes": note,
            }
        )

    if len(rows) < 2:
        abort(400, "Please enter at least two stops.")

    return rows


def write_rows_to_csv(rows) -> pathlib.Path:
    tmp_csv = OUTPUT_ROOT / f"upload_{uuid.uuid4().hex}.csv"
    with tmp_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["city", "country", "latitude", "longitude", "date", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return tmp_csv


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

    rows = build_rows_from_form(request.form)
    csv_path = write_rows_to_csv(rows)

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
