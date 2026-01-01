#!/usr/bin/env python3
"""Render a journey map from a CSV file and optionally export a camera tour video."""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

REQUIRED_COLUMNS = {"city", "country", "latitude", "longitude"}
OPTIONAL_COLUMNS = ["notes", "description"]
EARTH_RADIUS_KM = 6371.0


@dataclass
class GlobeFrame:
    lat: float
    lon: float
    scale: float
    seg_index: int = -1
    seg_t: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a journey visualization from a CSV file with city, country, latitude, "
            "and longitude columns. Optionally capture a camera flyover video."
        )
    )
    parser.add_argument(
        "csv_path",
        help="Path to the journey CSV file. Rows are rendered in order.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="journey_map.html",
        help="Path for the generated HTML visualization (default: journey_map.html).",
    )
    parser.add_argument(
        "--title",
        default="Around the World",
        help="Title displayed above the map.",
    )
    parser.add_argument(
        "--projection",
        default="orthographic",
        choices=[
            "orthographic",
            "natural earth",
            "equirectangular",
            "mercator",
        ],
        help="Projection used for the styled globe view.",
    )
    parser.add_argument(
        "--label-distance-km",
        type=float,
        default=60.0,
        help="Reserved for future spacing logic (currently unused).",
    )
    parser.add_argument(
        "--marker-size",
        type=int,
        default=26,
        help="Marker size (px). In auto label mode this may shrink for dense clusters.",
    )
    parser.add_argument(
        "--label-font-size",
        type=int,
        default=20,
        help="Font size for labels (px). In auto label mode this may shrink for dense clusters.",
    )
    parser.add_argument(
        "--label-offset-km",
        type=float,
        default=10.0,
        help="Vertical offset for stop titles in kilometers to avoid overlapping markers.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1100,
        help="Width of the figure in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=750,
        help="Height of the figure in pixels.",
    )
    parser.add_argument(
        "--video",
        help="Optional path for an MP4 video that flies the camera between stops.",
    )
    parser.add_argument(
        "--video-format",
        default="mp4",
        choices=["mp4", "webm", "mkv", "mov", "gif"],
        help="Container/format for exported video (default: mp4).",
    )
    parser.add_argument(
        "--bitrate",
        default="16M",
        help="Target video bitrate when using ffmpeg-backed formats (e.g., 8M, 16M).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second for the exported video (when --video is provided).",
    )
    parser.add_argument(
        "--linger",
        type=float,
        default=1.1,
        help="Seconds to linger when the camera reaches each stop in the video.",
    )
    parser.add_argument(
        "--zoom-boost",
        type=float,
        default=0.9,
        help=(
            "Extra zoom applied on segments to get closer at stops (globe scale * (1+boost*0.2))."
        ),
    )
    parser.add_argument(
        "--frame-format",
        default="png",
        choices=["png", "jpeg", "webp"],
        help="Image format for per-frame rendering during video export.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Upper bound on rendered frames for video export (0 to disable).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers for video frame rendering (0 to auto-tune).",
    )
    parser.add_argument(
        "--route-style",
        default="progressive",
        choices=["full", "progressive"],
        help="Draw the full route at once or progressively build it between stops.",
    )
    parser.add_argument(
        "--save-frames-dir",
        help="Optional directory to export individual video frames for inspection.",
    )
    return parser.parse_args()


def load_journey(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")

    for col in ("latitude", "longitude"):
        df[col] = pd.to_numeric(df[col], errors="raise")

    df = df.reset_index(drop=True)
    return df


def build_path_segments(df: pd.DataFrame) -> Tuple[List[float], List[float]]:
    lat_segments: List[float] = []
    lon_segments: List[float] = []
    for idx in range(len(df) - 1):
        lat_segments.extend([df.loc[idx, "latitude"], df.loc[idx + 1, "latitude"], None])
        lon_segments.extend([df.loc[idx, "longitude"], df.loc[idx + 1, "longitude"], None])
    return lat_segments, lon_segments


def build_path_segments_values(lats: List[float], lons: List[float]) -> Tuple[List[float], List[float]]:
    lat_segments: List[float] = []
    lon_segments: List[float] = []
    for idx in range(len(lats) - 1):
        lat_segments.extend([lats[idx], lats[idx + 1], None])
        lon_segments.extend([lons[idx], lons[idx + 1], None])
    return lat_segments, lon_segments


def build_partial_path_segments(
    df: pd.DataFrame,
    seg_index: int,
    seg_t: float,
) -> Tuple[List[float], List[float]]:
    """Build a polyline for the route up to the current segment progress.

    - Includes all fully completed segments < seg_index
    - Adds the current segment [seg_index -> seg_index+1] up to fraction seg_t (clamped)
    - If seg_index < 0, returns an empty path
    """
    if df.empty or seg_index < 0:
        return [], []
    n = len(df)
    if seg_index >= n - 1:
        # All segments completed: full path
        return build_path_segments(df)

    lat_segments: List[float] = []
    lon_segments: List[float] = []
    # Completed segments
    for idx in range(seg_index):
        lat_segments.extend([df.loc[idx, "latitude"], df.loc[idx + 1, "latitude"], None])
        lon_segments.extend([df.loc[idx, "longitude"], df.loc[idx + 1, "longitude"], None])
    # Partial current segment
    t = max(0.0, min(1.0, seg_t))
    start_lat = float(df.loc[seg_index, "latitude"]) 
    start_lon = float(df.loc[seg_index, "longitude"]) 
    end_lat = float(df.loc[seg_index + 1, "latitude"]) 
    end_lon = float(df.loc[seg_index + 1, "longitude"]) 
    cur_lat, cur_lon = interpolate_lat_lon((start_lat, start_lon), (end_lat, end_lon), t)
    lat_segments.extend([start_lat, cur_lat, None])
    lon_segments.extend([start_lon, cur_lon, None])
    return lat_segments, lon_segments


def build_partial_path_segments_values(
    lats: List[float],
    lons: List[float],
    seg_index: int,
    seg_t: float,
) -> Tuple[List[float], List[float]]:
    if not lats or seg_index < 0:
        return [], []
    n = len(lats)
    if seg_index >= n - 1:
        return build_path_segments_values(lats, lons)

    lat_segments: List[float] = []
    lon_segments: List[float] = []

    for idx in range(seg_index):
        lat_segments.extend([lats[idx], lats[idx + 1], None])
        lon_segments.extend([lons[idx], lons[idx + 1], None])

    t = max(0.0, min(1.0, seg_t))
    start_lat = float(lats[seg_index])
    start_lon = float(lons[seg_index])
    end_lat = float(lats[seg_index + 1])
    end_lon = float(lons[seg_index + 1])
    cur_lat, cur_lon = interpolate_lat_lon((start_lat, start_lon), (end_lat, end_lon), t)
    lat_segments.extend([start_lat, cur_lat, None])
    lon_segments.extend([start_lon, cur_lon, None])
    return lat_segments, lon_segments


def marker_numbers(df: pd.DataFrame) -> List[str]:
    return [str(idx + 1) for idx in range(len(df))]


def optional_value(row: pd.Series, key: str) -> Optional[str]:
    if key not in row:
        return None
    value = row[key]
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def stop_title(row: pd.Series) -> str:
    base = f"{row['city']}, {row['country']}"
    date_text = optional_value(row, "date")
    if date_text:
        return f"{base}, {date_text}"
    return base


def marker_hover_text(df: pd.DataFrame) -> List[str]:
    hover_lines: List[str] = []
    for _, row in df.iterrows():
        details: List[str] = [f"<b>{stop_title(row)}</b>"]
        for optional in OPTIONAL_COLUMNS:
            value = optional_value(row, optional)
            if value:
                label = optional.capitalize()
                details.append(f"{label}: {value}")
        hover_lines.append("<br>".join(details) + "<extra></extra>")
    return hover_lines


def label_text(df: pd.DataFrame) -> List[str]:
    labels: List[str] = []
    for idx, row in df.iterrows():
        labels.append(f"{idx + 1}. {stop_title(row)}")
    return labels


def offset_latitudes(lats: List[float], km: float) -> List[float]:
    """Return latitude values shifted northwards by km (approx 111 km per degree)."""
    if km == 0:
        return list(lats)
    delta = km / 111.0
    return [float(lat) + delta for lat in lats]


def min_interpoint_distance_km(df: pd.DataFrame) -> float:
    """Compute the minimum pairwise great-circle distance between points in km.

    Returns +inf if fewer than 2 points.
    """
    if len(df) < 2:
        return float("inf")
    min_d = float("inf")
    lats = df["latitude"].tolist()
    lons = df["longitude"].tolist()
    for i in range(len(lats) - 1):
        a = (lats[i], lons[i])
        for j in range(i + 1, len(lats)):
            b = (lats[j], lons[j])
            d = haversine_km(a, b)
            if d < min_d:
                min_d = d
    return min_d


def wrap_longitude(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def haversine_km(start: Tuple[float, float], end: Tuple[float, float]) -> float:
    lat1, lon1 = math.radians(start[0]), math.radians(start[1])
    lat2, lon2 = math.radians(end[0]), math.radians(end[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def interpolate_lat_lon(start: Tuple[float, float], end: Tuple[float, float], t: float) -> Tuple[float, float]:
    lat = start[0] + (end[0] - start[0]) * t
    lon0 = start[1]
    lon1 = end[1]
    delta = lon1 - lon0
    if abs(delta) > 180.0:
        if delta > 0:
            lon0 += 360.0
        else:
            lon1 += 360.0
        lon = lon0 + (lon1 - lon0) * t
    else:
        lon = lon0 + delta * t
    return lat, wrap_longitude(lon)


def normalize_bearing(delta: float) -> float:
    if delta > 180.0:
        delta -= 360.0
    elif delta < -180.0:
        delta += 360.0
    return delta


def segment_frame_count(distance_km: float) -> int:
    base = max(18, int(distance_km / 160.0) + 18)
    return min(base, 80)


def ease_in_out(t: float) -> float:
    """Smoothstep easing for less jittery motion (cubic ease-in-out)."""
    # Clamp for safety
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def scale_for_distance(distance_km: float) -> float:
    # Keep close hops readable without excessive zoom at each stop.
    if distance_km < 10:
        return 3.2
    if distance_km < 30:
        return 2.9
    if distance_km < 80:
        return 2.5
    if distance_km < 200:
        return 2.1
    if distance_km < 500:
        return 1.8
    if distance_km < 1200:
        return 1.5
    if distance_km < 2500:
        return 1.3
    return 1.1


def create_globe_frames(df: pd.DataFrame, fps: int, linger_seconds: float, zoom_boost: float = 0.0) -> List[GlobeFrame]:
    if df.empty:
        return []

    centers = list(zip(df["latitude"], df["longitude"]))
    linger_frames = max(1, int(fps * max(linger_seconds, 0.1)))
    frames: List[GlobeFrame] = []

    if len(centers) == 1:
        scale = 2.5
        for _ in range(linger_frames * 2):
            frames.append(GlobeFrame(lat=centers[0][0], lon=centers[0][1], scale=scale))
        return frames

    first_distance = haversine_km(centers[0], centers[1])
    base_scale = scale_for_distance(first_distance)
    for _ in range(linger_frames):
        frames.append(GlobeFrame(lat=centers[0][0], lon=centers[0][1], scale=base_scale, seg_index=-1, seg_t=0.0))

    current_rotation_lon = centers[0][1]
    prev_scale = base_scale
    for idx in range(len(centers) - 1):
        start = centers[idx]
        end = centers[idx + 1]
        distance = haversine_km(start, end)
        steps = segment_frame_count(distance)
        target_scale = scale_for_distance(distance) * (1.0 + max(0.0, zoom_boost) * 0.2)
        end_lon = end[1]
        lon_delta = normalize_bearing(end_lon - current_rotation_lon)
        if distance < 0.3:
            current_rotation_lon = current_rotation_lon + lon_delta
            prev_scale = target_scale
            for _ in range(linger_frames):
                frames.append(GlobeFrame(lat=end[0], lon=end[1], scale=prev_scale))
            continue
        for step in range(steps):
            t = (step + 1) / steps
            et = ease_in_out(t)
            lat, lon = interpolate_lat_lon(start, end, et)
            rotation_lon = current_rotation_lon + lon_delta * et
            scale = prev_scale + (target_scale - prev_scale) * et
            frames.append(GlobeFrame(lat=lat, lon=rotation_lon, scale=scale, seg_index=idx, seg_t=et))
        current_rotation_lon = current_rotation_lon + lon_delta
        prev_scale = target_scale
        # Linger without scaling out to avoid the pan-out effect
        for _ in range(linger_frames):
            frames.append(GlobeFrame(lat=end[0], lon=end[1], scale=prev_scale, seg_index=idx, seg_t=1.0))
    return frames


def limit_frame_defs(frame_defs: List[GlobeFrame], max_frames: int) -> List[GlobeFrame]:
    if max_frames <= 0 or len(frame_defs) <= max_frames:
        return frame_defs
    if max_frames == 1:
        return [frame_defs[-1]]
    step = (len(frame_defs) - 1) / (max_frames - 1)
    sampled = [frame_defs[int(i * step)] for i in range(max_frames)]
    sampled[-1] = frame_defs[-1]
    return sampled


def render_styled_geo(
    df: pd.DataFrame,
    *,
    title: str,
    projection: str,
    width: int,
    height: int,
    rotation: Optional[dict] = None,
    scale: Optional[float] = None,
    label_distance_km: float = 60.0,
    marker_size: int = 30,
    label_font_size: int = 20,
    progressive_route: bool = False,
    path_lat_override: Optional[List[float]] = None,
    path_lon_override: Optional[List[float]] = None,
    label_offset_km: float = 8.0,
    reached_upto: Optional[int] = None,
    include_hover: bool = True,
) -> go.Figure:
    path_lat, path_lon = build_path_segments(df)
    if path_lat_override is not None and path_lon_override is not None:
        path_lat, path_lon = path_lat_override, path_lon_override
    if progressive_route:
        path_lat, path_lon = [], []
    avg_lat = float(df["latitude"].mean()) if not df.empty else 0.0
    avg_lon = float(df["longitude"].mean()) if not df.empty else 0.0

    fig = go.Figure()

    # Always add the path trace; it may start empty for progressive mode
    fig.add_trace(
        go.Scattergeo(
            lat=path_lat,
            lon=path_lon,
            mode="lines",
            line=dict(color="#f25f3a", width=3),
            hoverinfo="skip",
            opacity=0.9,
        )
    )

    if not df.empty:
        # Always show markers and numbers; titles are conditionally revealed when reached
        local_marker = max(10, int(marker_size))
        local_label_font = max(9, int(label_font_size))
        fig.add_trace(
            go.Scattergeo(
                lat=df["latitude"],
                lon=df["longitude"],
                mode="markers",
                marker=dict(size=local_marker + 2, color="#ffd7b3", line=dict(width=2, color="#f25f3a")),
                **(
                    dict(hovertext=marker_hover_text(df), hoverinfo="text")
                    if include_hover
                    else dict(hoverinfo="skip")
                ),
            )
        )
        # Numbers overlay
        fig.add_trace(
            go.Scattergeo(
                lat=df["latitude"],
                lon=df["longitude"],
                mode="text",
                text=marker_numbers(df),
                textposition="middle center",
                textfont=dict(color="#4b2314", size=max(9, int(local_marker * 0.5))),
                hoverinfo="skip",
            )
        )
        # Titles overlay; if reached_upto is provided, hide unreached titles
        full_titles = label_text(df)
        if reached_upto is None:
            # Default: show all titles unless we're in progressive mode, where we start at the first stop
            active_upto = 0 if progressive_route else (len(df) - 1)
        else:
            active_upto = reached_upto
        titles = [t if i <= active_upto else "" for i, t in enumerate(full_titles)]
        fig.add_trace(
            go.Scattergeo(
                lat=offset_latitudes(df["latitude"].tolist(), label_offset_km),
                lon=df["longitude"],
                mode="text",
                text=titles,
                textposition="top center",
                textfont=dict(color="#fdf2e9", size=local_label_font),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        showlegend=False,
        width=width,
        height=height,
        autosize=False,
        font=dict(family="Space Grotesk, Avenir Next, Segoe UI, DejaVu Sans, sans-serif"),
        paper_bgcolor="#0b1c2b",
        margin=dict(l=0, r=0, t=0, b=0),
        geo=dict(
            projection=dict(
                type=projection,
                rotation=rotation or dict(lat=avg_lat, lon=avg_lon),
                scale=scale or 0.95,
            ),
            domain=dict(x=[0, 1], y=[0, 1]),
            showland=True,
            landcolor="#1f4e5a",
            showcountries=True,
            countrycolor="#dfe9ec",
            showocean=True,
            oceancolor="#0b2a3a",
            showcoastlines=True,
            coastlinecolor="#6fa9b3",
            showframe=False,
            bgcolor="#0b1c2b",
        ),
    )

    # Title inside the map as an overlay annotation
    if title:
        fig.add_annotation(
            x=0.5,
            y=0.98,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="top",
            text=title,
            showarrow=False,
            font=dict(size=26, color="#fdf2e9"),
            bgcolor="rgba(11, 28, 43, 0.55)",
            borderpad=6,
        )

    # Progressive frames: one per stop, route grows cumulatively
    if progressive_route and len(df) > 1:
        frames = []
        for upto in range(0, len(df)):
            part_df = df.iloc[: upto + 1].reset_index(drop=True)
            f_lat, f_lon = build_path_segments(part_df)
            titles = [t if i <= upto else "" for i, t in enumerate(label_text(df))]
            # traces: 0 path, 1 markers, 2 numbers, 3 titles
            frames.append(
                go.Frame(
                    name=f"step-{upto}",
                    data=[
                        go.Scattergeo(lat=f_lat, lon=f_lon),
                        go.Scattergeo(text=titles),
                    ],
                    traces=[0, 3],
                )
            )
        fig.frames = frames
        # Slider + play button
        steps = [
            dict(
                method="animate",
                args=[[f"step-{i}"], {"frame": {"duration": 600, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                label=str(i + 1),
            )
            for i in range(0, len(df))
        ]
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0.9,
                    yanchor="top",
                    x=0.5,
                    xanchor="center",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 600, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    y=0.06,
                    x=0.1,
                    len=0.8,
                    pad={"b": 10, "t": 50},
                    active=0,
                    steps=steps,
                )
            ],
        )

    return fig


def ensure_kaleido_available() -> None:
    try:
        import importlib
        importlib.import_module("kaleido")
    except ImportError as exc:  # pragma: no cover - purely defensive
        raise RuntimeError(
            "Video export requires kaleido. Install it with 'pip install kaleido'."
        ) from exc


def figure_to_image(fig: go.Figure, *, img_format: str = "png", width: int, height: int) -> bytes:
    # Plotly/kaleido supports png, jpeg, webp, svg (we use raster formats)
    # Use scale=1 explicitly to avoid devicePixelRatio variance across renders.
    return fig.to_image(format=img_format, width=width, height=height, scale=1, validate=False)


def inject_fonts(html: str) -> str:
    font_block = (
        "<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">"
        "<link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>"
        "<link href=\"https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap\" "
        "rel=\"stylesheet\">"
        "<style>html, body { margin: 0; padding: 0; }"
        ".js-plotly-plot, .plotly, .main-svg { font-family: 'Space Grotesk', 'Avenir Next', "
        "'Segoe UI', 'DejaVu Sans', sans-serif; }"
        "</style>"
    )
    if "<head>" not in html:
        return html
    return html.replace("<head>", f"<head>{font_block}", 1)


def start_kaleido_sync_server() -> None:
    """Start Kaleido's persistent sync server (Kaleido v1) to avoid per-frame browser spin-up."""
    try:
        import kaleido  # type: ignore
    except ImportError:
        return
    start = getattr(kaleido, "start_sync_server", None)
    if start is None:
        return
    try:
        # silence_warnings avoids noisy "already running" messages on repeated calls.
        start(silence_warnings=True)
    except Exception:
        # If the server fails to start (missing chrome, sandbox restrictions, etc.),
        # plotly will fall back to one-shot rendering and raise if unusable.
        return


def stop_kaleido_sync_server() -> None:
    try:
        import kaleido  # type: ignore
    except ImportError:
        return
    stop = getattr(kaleido, "stop_sync_server", None)
    if stop is None:
        return
    try:
        stop()
    except Exception:
        return


class _VideoFrameWorkerState:
    def __init__(
        self,
        *,
        lats: List[float],
        lons: List[float],
        full_titles: List[str],
        fig: go.Figure,
        path_trace: Any,
        title_trace: Any,
        geo: Any,
        width: int,
        height: int,
        frame_format: str,
    ) -> None:
        self.lats = lats
        self.lons = lons
        self.full_titles = full_titles
        self.fig = fig
        self.path_trace = path_trace
        self.title_trace = title_trace
        self.geo = geo
        self.width = width
        self.height = height
        self.frame_format = frame_format


_VIDEO_FRAME_WORKER_STATE: Optional[_VideoFrameWorkerState] = None


def _reached_upto(frame: GlobeFrame) -> int:
    return 0 if frame.seg_index < 0 else frame.seg_index + (1 if frame.seg_t >= 1.0 else 0)


def _init_video_frame_worker(config: Dict[str, Any]) -> None:
    global _VIDEO_FRAME_WORKER_STATE
    start_kaleido_sync_server()

    df = load_journey(Path(config["csv_path"]))
    lats = df["latitude"].tolist()
    lons = df["longitude"].tolist()
    full_titles = label_text(df)

    first_frame = config["first_frame"]
    p_lat, p_lon = build_partial_path_segments_values(lats, lons, first_frame.seg_index, first_frame.seg_t)
    fig = render_styled_geo(
        df,
        title=config["title"],
        projection=config["projection"],
        width=config["width"],
        height=config["height"],
        rotation=dict(lat=first_frame.lat, lon=first_frame.lon),
        scale=first_frame.scale,
        label_distance_km=config["label_distance_km"],
        marker_size=config["marker_size"],
        label_font_size=config["label_font_size"],
        path_lat_override=p_lat,
        path_lon_override=p_lon,
        label_offset_km=config["label_offset_km"],
        reached_upto=_reached_upto(first_frame),
        include_hover=True,
    )

    path_trace = fig.data[0] if fig.data else None
    title_trace = fig.data[3] if len(fig.data) > 3 else None
    geo = fig.layout.geo
    _VIDEO_FRAME_WORKER_STATE = _VideoFrameWorkerState(
        lats=lats,
        lons=lons,
        full_titles=full_titles,
        fig=fig,
        path_trace=path_trace,
        title_trace=title_trace,
        geo=geo,
        width=int(config["width"]),
        height=int(config["height"]),
        frame_format=str(config["frame_format"]),
    )


def _render_video_frame_bytes(frame: GlobeFrame) -> bytes:
    state = _VIDEO_FRAME_WORKER_STATE
    if state is None:
        raise RuntimeError("Video frame worker not initialized.")
    p_lat, p_lon = build_partial_path_segments_values(state.lats, state.lons, frame.seg_index, frame.seg_t)
    if state.path_trace is not None:
        state.path_trace.lat = p_lat
        state.path_trace.lon = p_lon
    if state.title_trace is not None:
        upto = _reached_upto(frame)
        state.title_trace.text = [t if i <= upto else "" for i, t in enumerate(state.full_titles)]
    state.geo.projection.rotation = dict(lat=frame.lat, lon=frame.lon)
    state.geo.projection.scale = frame.scale
    return figure_to_image(
        state.fig,
        img_format=state.frame_format,
        width=state.width,
        height=state.height,
    )


def write_video(
    df: pd.DataFrame,
    *,
    args: argparse.Namespace,
) -> None:
    ensure_kaleido_available()
    try:
        import imageio.v3 as iio
        import imageio
    except ImportError as exc:
        raise RuntimeError(
            "Video export requires imageio. Install it with 'pip install imageio imageio-ffmpeg'."
        ) from exc

    output_path = Path(args.video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_frames_dir:
        os.makedirs(args.save_frames_dir, exist_ok=True)

    frame_defs = create_globe_frames(df, args.fps, args.linger, args.zoom_boost)
    if not frame_defs:
        raise RuntimeError("Cannot build video: the CSV has no locations.")
    original_count = len(frame_defs)
    frame_defs = limit_frame_defs(frame_defs, args.max_frames)
    if len(frame_defs) < original_count:
        print(f"Info: reducing frames from {original_count} to {len(frame_defs)} (max_frames={args.max_frames}).")

    total_frames = len(frame_defs)
    workers = int(args.workers or 0)
    if workers <= 0:
        cpu = os.cpu_count() or 1
        workers = 1 if total_frames < 200 else min(4, cpu)
    workers = max(1, workers)
    if workers > 1:
        print(f"Info: rendering frames with {workers} workers.")

    # if args.save_frames_dir:
    #     frame_dir = Path(args.save_frames_dir)
    #     frame_dir.mkdir(parents=True, exist_ok=True)
    #     try:
    #         from PIL import Image  # type: ignore
    #     except ImportError as exc:  # pragma: no cover - depends on optional deps
    #         raise RuntimeError(
    #             "Saving frames requires Pillow. Install it with 'pip install Pillow'."
    #         ) from exc
    #     for idx, frame in enumerate(rendered_frames):
    #         img = Image.fromarray(frame, mode="RGB")
    #         img.save(frame_dir / f"{preferred_mode}_frame_{idx:05d}.png")

    # Choose container/codec and align output extension
    chosen_fmt = (args.video_format or "mp4").lower()
    if output_path.suffix.lower().lstrip(".") != chosen_fmt:
        output_path = output_path.with_suffix(f".{chosen_fmt}")
        print(f"ℹ️  Adjusted output extension to match --video-format: {output_path.name}")

    codec_by_fmt = {
        "mp4": "libx264",
        "mkv": "libx264",
        "mov": "libx264",
        "webm": "libvpx-vp9",
    }

    try:
        marker_size = max(12, int(args.marker_size * 0.9))
        label_font_size = max(10, int(args.label_font_size * 0.9))
        ext = ".jpg" if args.frame_format == "jpeg" else f".{args.frame_format}"

        def render_and_append(append_data) -> None:
            prev_frame = None
            glitch_threshold_pct = 33.0
            sample_target = 200

            def maybe_append_frame(image_bytes: bytes) -> None:
                nonlocal prev_frame
                rendered_frame = iio.imread(image_bytes, extension=ext)
                if prev_frame is not None:
                    if rendered_frame.shape != prev_frame.shape:
                        print("Significant difference detected")
                        return
                    height, width = rendered_frame.shape[:2]
                    stride = max(1, int(max(height, width) / sample_target))
                    sample_frame = rendered_frame[::stride, ::stride]
                    prev_sample = prev_frame[::stride, ::stride]
                    sample_diff = (sample_frame != prev_sample).sum()
                    sample_pct = sample_diff / sample_frame.size * 100
                    if sample_pct > glitch_threshold_pct:
                        full_diff = (rendered_frame != prev_frame).sum()
                        full_pct = full_diff / rendered_frame.size * 100
                        if full_pct > glitch_threshold_pct:
                            print("Significant difference detected")
                            return
                append_data(rendered_frame)
                prev_frame = rendered_frame

            if workers == 1:
                start_kaleido_sync_server()
                try:
                    lats = df["latitude"].tolist()
                    lons = df["longitude"].tolist()
                    full_titles = label_text(df)

                    first_frame = frame_defs[0]
                    p_lat, p_lon = build_partial_path_segments_values(lats, lons, first_frame.seg_index, first_frame.seg_t)
                    fig = render_styled_geo(
                        df,
                        title=args.title,
                        projection=args.projection,
                        width=args.width,
                        height=args.height,
                        rotation=dict(lat=first_frame.lat, lon=first_frame.lon),
                        scale=first_frame.scale,
                        label_distance_km=args.label_distance_km,
                        marker_size=marker_size,
                        label_font_size=label_font_size,
                        path_lat_override=p_lat,
                        path_lon_override=p_lon,
                        label_offset_km=args.label_offset_km,
                        reached_upto=_reached_upto(first_frame),
                        include_hover=True,
                    )

                    path_trace = fig.data[0] if fig.data else None
                    title_trace = fig.data[3] if len(fig.data) > 3 else None
                    geo = fig.layout.geo

                    for frame_num, frame in enumerate(frame_defs):
                        if frame_num:
                            p_lat, p_lon = build_partial_path_segments_values(lats, lons, frame.seg_index, frame.seg_t)
                            if path_trace is not None:
                                path_trace.lat = p_lat
                                path_trace.lon = p_lon
                            if title_trace is not None:
                                upto = _reached_upto(frame)
                                title_trace.text = [t if i <= upto else "" for i, t in enumerate(full_titles)]
                            geo.projection.rotation = dict(lat=frame.lat, lon=frame.lon)
                            geo.projection.scale = frame.scale

                        if total_frames <= 20 or frame_num % 10 == 0 or frame_num == total_frames - 1:
                            print(f"Rendered frame {frame_num + 1}/{total_frames}")

                        image_bytes = figure_to_image(
                            fig,
                            img_format=args.frame_format,
                            width=args.width,
                            height=args.height,
                        )

                        if args.save_frames_dir:
                            with open(os.path.join(args.save_frames_dir, f"frame_{frame_num:05d}{ext}"), "wb") as handle:
                                handle.write(image_bytes)

                        maybe_append_frame(image_bytes)
                finally:
                    stop_kaleido_sync_server()
            else:
                import multiprocessing as mp

                config: Dict[str, Any] = dict(
                    csv_path=str(args.csv_path),
                    title=str(args.title),
                    projection=str(args.projection),
                    width=int(args.width),
                    height=int(args.height),
                    label_distance_km=float(args.label_distance_km),
                    marker_size=marker_size,
                    label_font_size=label_font_size,
                    label_offset_km=float(args.label_offset_km),
                    frame_format=str(args.frame_format),
                    first_frame=frame_defs[0],
                )

                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=workers, initializer=_init_video_frame_worker, initargs=(config,)) as pool:
                    for frame_num, image_bytes in enumerate(pool.imap(_render_video_frame_bytes, frame_defs, chunksize=2)):
                        if total_frames <= 20 or frame_num % 10 == 0 or frame_num == total_frames - 1:
                            print(f"Rendered frame {frame_num + 1}/{total_frames}")

                        if args.save_frames_dir:
                            with open(os.path.join(args.save_frames_dir, f"frame_{frame_num:05d}{ext}"), "wb") as handle:
                                handle.write(image_bytes)

                        maybe_append_frame(image_bytes)

        if chosen_fmt not in codec_by_fmt and chosen_fmt != "gif":
            output_path = output_path.with_suffix(".mp4")
            chosen_fmt = "mp4"
            print(f"ℹ️  Unknown --video-format; falling back to {output_path.name}")

        if chosen_fmt in codec_by_fmt:
            write_kwargs: Dict[str, Any] = dict(
                fps=args.fps,
                codec=codec_by_fmt[chosen_fmt],
                bitrate=args.bitrate,
                macro_block_size=1,
            )
            try:
                writer = imageio.get_writer(output_path, **write_kwargs)
            except TypeError:
                write_kwargs.pop("macro_block_size", None)
                writer = imageio.get_writer(output_path, **write_kwargs)
        else:
            writer = imageio.get_writer(output_path, fps=args.fps, loop=0)

        try:
            render_and_append(writer.append_data)
        finally:
            writer.close()
    except RuntimeError as exc:  # pragma: no cover - depends on local codecs
        raise RuntimeError(
            "imageio could not find a working ffmpeg/codec. Install 'imageio-ffmpeg' and ensure ffmpeg is available."
        ) from exc

    print(f"🎬 Journey video written to {output_path.resolve()}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = load_journey(csv_path)

    fig = render_styled_geo(
        df,
        title=args.title,
        projection=args.projection,
        width=args.width,
        height=args.height,
        label_distance_km=args.label_distance_km,
        marker_size=args.marker_size,
        label_font_size=args.label_font_size,
        progressive_route=(args.route_style == "progressive"),
        label_offset_km=args.label_offset_km,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    html = inject_fonts(html)
    output_path.write_text(html, encoding="utf-8")
    print(f"✨ Journey map written to {output_path.resolve()}")

    if args.video:
        write_video(
            df,
            args=args,
        )


if __name__ == "__main__":
    main()
