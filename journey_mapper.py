#!/usr/bin/env python3
"""Render a journey map from a CSV file and optionally export a camera tour video."""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

REQUIRED_COLUMNS = {"city", "country", "latitude", "longitude"}
OPTIONAL_COLUMNS = ["date", "notes", "description"]
EARTH_RADIUS_KM = 6371.0


@dataclass
class CameraFrame:
    lat: float
    lon: float
    zoom: float
    bearing: float
    pitch: float = 45.0


@dataclass
class GlobeFrame:
    lat: float
    lon: float
    scale: float


class MapboxTileUnavailable(RuntimeError):
    """Raised when static satellite rendering fails due to unavailable Mapbox tiles."""


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
        help="Projection used when the styled (non-satellite) map is active.",
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
        "--map-style",
        default="styled",
        choices=["styled", "satellite"],
        help="Choose between the original styled globe or a satellite basemap (requires Mapbox token).",
    )
    parser.add_argument(
        "--mapbox-token",
        help="Mapbox access token required for satellite basemap. Defaults to MAPBOX_TOKEN env var if omitted.",
    )
    parser.add_argument(
        "--video",
        help="Optional path for an MP4 video that flies the camera between stops.",
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
        default=0.8,
        help="Seconds to linger when the camera reaches each stop in the video.",
    )
    parser.add_argument(
        "--video-map-style",
        default="auto",
        choices=["auto", "styled", "satellite"],
        help="Basemap preference for video export: auto (match map if possible), styled globe, or satellite.",
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


def marker_numbers(df: pd.DataFrame) -> List[str]:
    return [str(idx + 1) for idx in range(len(df))]


def marker_hover_text(df: pd.DataFrame) -> List[str]:
    hover_lines: List[str] = []
    for _, row in df.iterrows():
        details: List[str] = [f"<b>{row['city']}, {row['country']}</b>"]
        for optional in OPTIONAL_COLUMNS:
            if optional in row and pd.notna(row[optional]):
                label = optional.capitalize()
                details.append(f"{label}: {row[optional]}")
        hover_lines.append("<br>".join(details) + "<extra></extra>")
    return hover_lines


def label_text(df: pd.DataFrame) -> List[str]:
    labels: List[str] = []
    for idx, row in df.iterrows():
        labels.append(f"{idx + 1}. {row['city']}, {row['country']}")
    return labels


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


def initial_bearing(start: Tuple[float, float], end: Tuple[float, float]) -> float:
    lat1, lon1 = math.radians(start[0]), math.radians(start[1])
    lat2, lon2 = math.radians(end[0]), math.radians(end[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return ((bearing + 180.0) % 360.0) - 180.0


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


def zoom_for_distance(distance_km: float) -> float:
    if distance_km < 40:
        return 7.4
    if distance_km < 120:
        return 6.2
    if distance_km < 300:
        return 5.1
    if distance_km < 800:
        return 4.0
    if distance_km < 1600:
        return 3.0
    if distance_km < 3200:
        return 2.1
    return 1.3


def scale_for_distance(distance_km: float) -> float:
    if distance_km < 40:
        return 3.6
    if distance_km < 120:
        return 2.8
    if distance_km < 300:
        return 2.1
    if distance_km < 800:
        return 1.6
    if distance_km < 1600:
        return 1.2
    return 1.0


def create_mapbox_camera_frames(
    df: pd.DataFrame, fps: int, linger_seconds: float
) -> List[CameraFrame]:
    if df.empty:
        return []

    centers = list(zip(df["latitude"], df["longitude"]))
    linger_frames = max(1, int(fps * max(linger_seconds, 0.1)))
    frames: List[CameraFrame] = []

    if len(centers) == 1:
        zoom = 6.0
        bearing = 0.0
        for _ in range(linger_frames * 2):
            frames.append(CameraFrame(lat=centers[0][0], lon=centers[0][1], zoom=zoom, bearing=bearing))
        return frames

    first_segment_distance = haversine_km(centers[0], centers[1])
    base_zoom = zoom_for_distance(first_segment_distance)
    base_bearing = initial_bearing(centers[0], centers[1])
    for _ in range(linger_frames):
        frames.append(CameraFrame(lat=centers[0][0], lon=centers[0][1], zoom=base_zoom, bearing=base_bearing))

    prev_bearing = base_bearing
    for idx in range(len(centers) - 1):
        start = centers[idx]
        end = centers[idx + 1]
        distance = haversine_km(start, end)
        steps = segment_frame_count(distance)
        target_zoom = zoom_for_distance(distance)
        if math.isclose(distance, 0.0, abs_tol=1e-6):
            bearing_delta = 45.0
        else:
            target_bearing = initial_bearing(start, end)
            bearing_delta = normalize_bearing(target_bearing - prev_bearing)
        if distance < 120:
            bearing_delta = 30.0 if bearing_delta >= 0 else -30.0
        for step in range(steps):
            t = (step + 1) / steps
            lat, lon = interpolate_lat_lon(start, end, t)
            bearing = prev_bearing + bearing_delta * t
            frames.append(CameraFrame(lat=lat, lon=lon, zoom=target_zoom, bearing=bearing))
        prev_bearing = prev_bearing + bearing_delta
        end_zoom = zoom_for_distance(distance) * 0.92
        for _ in range(linger_frames):
            frames.append(CameraFrame(lat=end[0], lon=end[1], zoom=end_zoom, bearing=prev_bearing))
    return frames


def create_globe_frames(df: pd.DataFrame, fps: int, linger_seconds: float) -> List[GlobeFrame]:
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
        frames.append(GlobeFrame(lat=centers[0][0], lon=centers[0][1], scale=base_scale))

    current_rotation_lon = centers[0][1]
    for idx in range(len(centers) - 1):
        start = centers[idx]
        end = centers[idx + 1]
        distance = haversine_km(start, end)
        steps = segment_frame_count(distance)
        target_scale = scale_for_distance(distance)
        end_lon = end[1]
        lon_delta = normalize_bearing(end_lon - current_rotation_lon)
        for step in range(steps):
            t = (step + 1) / steps
            lat, lon = interpolate_lat_lon(start, end, t)
            rotation_lon = current_rotation_lon + lon_delta * t
            frames.append(GlobeFrame(lat=lat, lon=rotation_lon, scale=target_scale))
        current_rotation_lon = current_rotation_lon + lon_delta
        end_scale = scale_for_distance(distance) * 0.92
        for _ in range(linger_frames):
            frames.append(GlobeFrame(lat=end[0], lon=end[1], scale=end_scale))
    return frames


def render_styled_geo(
    df: pd.DataFrame,
    *,
    title: str,
    projection: str,
    width: int,
    height: int,
    rotation: Optional[dict] = None,
    scale: Optional[float] = None,
) -> go.Figure:
    path_lat, path_lon = build_path_segments(df)
    avg_lat = float(df["latitude"].mean()) if not df.empty else 0.0
    avg_lon = float(df["longitude"].mean()) if not df.empty else 0.0

    fig = go.Figure()

    if path_lat and path_lon:
        fig.add_trace(
            go.Scattergeo(
                lat=path_lat,
                lon=path_lon,
                mode="lines",
                line=dict(color="#00d1ff", width=3),
                hoverinfo="skip",
                opacity=0.9,
            )
        )

    if not df.empty:
        fig.add_trace(
            go.Scattergeo(
                lat=df["latitude"],
                lon=df["longitude"],
                mode="markers+text",
                marker=dict(size=28, color="#ff9f1c", line=dict(width=2, color="#2b2d42")),
                text=marker_numbers(df),
                textposition="middle center",
                textfont=dict(color="#001219", size=14),
                hovertext=marker_hover_text(df),
                hoverinfo="text",
            )
        )
        fig.add_trace(
            go.Scattergeo(
                lat=df["latitude"],
                lon=df["longitude"],
                mode="text",
                text=label_text(df),
                textposition="top center",
                textfont=dict(color="#edf2f4", size=13),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=30, color="#edf2f4")),
        showlegend=False,
        width=width,
        height=height,
        paper_bgcolor="#0b132b",
        margin=dict(l=10, r=10, t=60, b=10),
        geo=dict(
            projection=dict(
                type=projection,
                rotation=rotation or dict(lat=avg_lat, lon=avg_lon),
                scale=scale or 0.95,
            ),
            showland=True,
            landcolor="#1d3557",
            showcountries=True,
            countrycolor="#f1faee",
            showocean=True,
            oceancolor="#073b4c",
            showcoastlines=True,
            coastlinecolor="#118ab2",
            showframe=False,
            bgcolor="#0b132b",
        ),
    )

    return fig


def render_satellite_map(
    df: pd.DataFrame,
    *,
    title: str,
    width: int,
    height: int,
    token: str,
    center: Optional[Tuple[float, float]] = None,
    zoom: Optional[float] = None,
    bearing: float = 0.0,
    pitch: float = 45.0,
) -> go.Figure:
    path_lat, path_lon = build_path_segments(df)
    avg_lat = float(df["latitude"].mean()) if not df.empty else 0.0
    avg_lon = float(df["longitude"].mean()) if not df.empty else 0.0
    center_lat, center_lon = center or (avg_lat, avg_lon)

    fig = go.Figure()

    if path_lat and path_lon:
        fig.add_trace(
            go.Scattermapbox(
                lat=path_lat,
                lon=path_lon,
                mode="lines",
                line=dict(color="#16f4d0", width=4),
                hoverinfo="skip",
                opacity=0.8,
            )
        )

    if not df.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=df["latitude"],
                lon=df["longitude"],
                mode="markers",
                marker=dict(size=32, color="#023047", opacity=0.9),
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scattermapbox(
                lat=df["latitude"],
                lon=df["longitude"],
                mode="markers+text",
                marker=dict(size=26, color="#ffb703", opacity=0.95),
                text=marker_numbers(df),
                textposition="middle center",
                textfont=dict(color="#001219", size=13),
                hovertext=marker_hover_text(df),
                hoverinfo="text",
            )
        )
        fig.add_trace(
            go.Scattermapbox(
                lat=df["latitude"],
                lon=df["longitude"],
                mode="text",
                text=label_text(df),
                textposition="top center",
                textfont=dict(color="#f7f9fb", size=12),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=30, color="#f7f9fb")),
        showlegend=False,
        width=width,
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="#000814",
        mapbox=dict(
            accesstoken=token,
            style="satellite-streets",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom if zoom is not None else 1.4,
            bearing=bearing,
            pitch=pitch,
        ),
    )

    return fig


def ensure_kaleido_available() -> None:
    try:
        import kaleido  # noqa: F401
    except ImportError as exc:  # pragma: no cover - purely defensive
        raise RuntimeError(
            "Video export requires kaleido. Install it with 'pip install kaleido'."
        ) from exc


def figure_to_png(fig: go.Figure, *, width: int, height: int) -> bytes:
    try:
        return fig.to_image(format="png", width=width, height=height)
    except Exception as exc:  # pragma: no cover - depends on local kaleido runtime
        if "Mapbox error" in str(exc):
            raise MapboxTileUnavailable("Mapbox tiles unavailable for static rendering.") from exc
        raise


def write_video(
    df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    preferred_mode: str,
    token: Optional[str],
) -> None:
    ensure_kaleido_available()
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError(
            "Video export requires imageio. Install it with 'pip install imageio imageio-ffmpeg'."
        ) from exc

    output_path = Path(args.video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def render_mode_frames(mode: str) -> List:
        if mode == "mapbox":
            frame_defs = create_mapbox_camera_frames(df, args.fps, args.linger)
            if not frame_defs:
                raise RuntimeError("Cannot build video: the CSV has no locations.")
            rendered: List = []
            for frame in frame_defs:
                fig = render_satellite_map(
                    df,
                    title=args.title,
                    width=args.width,
                    height=args.height,
                    token=token or "",
                    center=(frame.lat, frame.lon),
                    zoom=frame.zoom,
                    bearing=frame.bearing,
                    pitch=frame.pitch,
                )
                image_bytes = figure_to_png(fig, width=args.width, height=args.height)
                rendered.append(iio.imread(image_bytes, extension=".png"))
            return rendered

        frame_defs = create_globe_frames(df, args.fps, args.linger)
        if not frame_defs:
            raise RuntimeError("Cannot build video: the CSV has no locations.")
        rendered: List = []
        for frame in frame_defs:
            fig = render_styled_geo(
                df,
                title=args.title,
                projection=args.projection,
                width=args.width,
                height=args.height,
                rotation=dict(lat=frame.lat, lon=frame.lon),
                scale=frame.scale,
            )
            image_bytes = figure_to_png(fig, width=args.width, height=args.height)
            rendered.append(iio.imread(image_bytes, extension=".png"))
        return rendered

    try:
        rendered_frames = render_mode_frames(preferred_mode)
    except MapboxTileUnavailable:
        if preferred_mode == "mapbox":
            print("⚠️  Mapbox tiles unavailable during video capture; falling back to the styled globe for MP4 output.")
            rendered_frames = render_mode_frames("styled")
        else:
            raise

    try:
        iio.imwrite(
            output_path,
            rendered_frames,
            fps=args.fps,
            codec="libx264",
            bitrate="16M",
        )
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

    token = args.mapbox_token or os.environ.get("MAPBOX_TOKEN")
    use_mapbox = args.map_style == "satellite" and token
    if args.map_style == "satellite" and not token:
        print("⚠️  Mapbox token not found; falling back to the styled globe basemap.")

    if use_mapbox:
        fig = render_satellite_map(
            df,
            title=args.title,
            width=args.width,
            height=args.height,
            token=token or "",
        )
    else:
        fig = render_styled_geo(
            df,
            title=args.title,
            projection=args.projection,
            width=args.width,
            height=args.height,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"✨ Journey map written to {output_path.resolve()}")

    if args.video:
        video_preference = args.video_map_style
        token_available = bool(token)
        if video_preference == "satellite":
            if not token_available:
                print("⚠️  Satellite video requested but no Mapbox token found; using styled globe instead.")
                preferred_mode = "styled"
            else:
                preferred_mode = "mapbox"
        elif video_preference == "styled":
            preferred_mode = "styled"
        else:  # auto
            preferred_mode = "mapbox" if token_available else "styled"

        write_video(
            df,
            args=args,
            preferred_mode=preferred_mode,
            token=token,
        )


if __name__ == "__main__":
    main()
