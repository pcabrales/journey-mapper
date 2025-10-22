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
    seg_index: int = -1
    seg_t: float = 0.0


@dataclass
class GlobeFrame:
    lat: float
    lon: float
    scale: float
    seg_index: int = -1
    seg_t: float = 0.0


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
        "--map-style",
        default="styled",
        choices=["styled", "satellite"],
        help="Choose between the original styled globe or a satellite basemap (requires Mapbox token).",
    )
    parser.add_argument(
        "--mapbox-style",
        default="satellite",
        choices=["satellite", "satellite-streets", "outdoors", "light", "dark"],
        help="Mapbox style for satellite mode.",
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
            "Extra zoom applied on segments to get closer at stops (mapbox zoom +boost, globe scale * (1+boost*0.2))."
        ),
    )
    parser.add_argument(
        "--video-map-style",
        default="auto",
        choices=["auto", "styled", "satellite"],
        help="Basemap preference for video export: auto (match map if possible), styled globe, or satellite.",
    )
    parser.add_argument(
        "--frame-format",
        default="png",
        choices=["png", "jpeg", "webp"],
        help="Image format for per-frame rendering during video export.",
    )
    parser.add_argument(
        "--route-style",
        default="progressive",
        choices=["full", "progressive"],
        help="Draw the full route at once or progressively build it between stops.",
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


def marker_numbers(df: pd.DataFrame) -> List[str]:
    return [str(idx + 1) for idx in range(len(df))]


def marker_hover_text(df: pd.DataFrame) -> List[str]:
    hover_lines: List[str] = []
    for _, row in df.iterrows():
        details: List[str] = [f"<b>{row['city']}, {row['country']}, {row['date']}</b>"]
        for optional in OPTIONAL_COLUMNS:
            if optional in row and pd.notna(row[optional]):
                label = optional.capitalize()
                details.append(f"{label}: {row[optional]}")
        hover_lines.append("<br>".join(details) + "<extra></extra>")
    return hover_lines


def label_text(df: pd.DataFrame) -> List[str]:
    labels: List[str] = []
    for idx, row in df.iterrows():
        labels.append(f"{idx + 1}. {row['city']}, {row['country']}, {row['date']}")
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
    # Closer default zooms for dense city hops
    if distance_km < 5:
        return 13.0
    if distance_km < 15:
        return 11.0
    if distance_km < 40:
        return 9.0
    if distance_km < 120:
        return 7.2
    if distance_km < 300:
        return 5.8
    if distance_km < 800:
        return 4.4
    if distance_km < 1600:
        return 3.2
    if distance_km < 3200:
        return 2.1
    return 1.3


def ease_in_out(t: float) -> float:
    """Smoothstep easing for less jittery motion (cubic ease-in-out)."""
    # Clamp for safety
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def scale_for_distance(distance_km: float) -> float:
    # Stronger zoom-in for close segments on the styled globe
    if distance_km < 5:
        return 6.0
    if distance_km < 15:
        return 5.2
    if distance_km < 40:
        return 4.5
    if distance_km < 120:
        return 3.4
    if distance_km < 300:
        return 2.6
    if distance_km < 800:
        return 1.8
    if distance_km < 1600:
        return 1.3
    return 1.0


def create_mapbox_camera_frames(
    df: pd.DataFrame, fps: int, linger_seconds: float, zoom_boost: float = 0.0
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
        frames.append(CameraFrame(lat=centers[0][0], lon=centers[0][1], zoom=base_zoom, bearing=base_bearing, seg_index=-1, seg_t=0.0))

    prev_bearing = base_bearing
    prev_zoom = base_zoom
    for idx in range(len(centers) - 1):
        start = centers[idx]
        end = centers[idx + 1]
        distance = haversine_km(start, end)
        steps = segment_frame_count(distance)
        target_zoom = zoom_for_distance(distance) + max(0.0, zoom_boost)
        if math.isclose(distance, 0.0, abs_tol=1e-6):
            bearing_delta = 0.0
        else:
            target_bearing = initial_bearing(start, end)
            bearing_delta = normalize_bearing(target_bearing - prev_bearing)
        if distance < 8:
            bearing_delta = 0.0
        elif distance < 40:
            bearing_delta = 15.0 if bearing_delta >= 0 else -15.0
        elif distance < 120:
            # Keep small hops snappy but not abrupt
            bearing_delta = 20.0 if bearing_delta >= 0 else -20.0
        # If effectively same point, just linger
        if distance < 0.3:
            prev_bearing = prev_bearing + bearing_delta
            prev_zoom = target_zoom
            for _ in range(linger_frames):
                frames.append(CameraFrame(lat=end[0], lon=end[1], zoom=prev_zoom, bearing=prev_bearing))
            continue
        for step in range(steps):
            t = (step + 1) / steps
            et = ease_in_out(t)
            lat, lon = interpolate_lat_lon(start, end, et)
            bearing = prev_bearing + bearing_delta * et
            zoom = prev_zoom + (target_zoom - prev_zoom) * et
            frames.append(CameraFrame(lat=lat, lon=lon, zoom=zoom, bearing=bearing, seg_index=idx, seg_t=et))
        prev_bearing = prev_bearing + bearing_delta
        prev_zoom = target_zoom
        # Linger without zooming out to avoid the pan-out effect
        for _ in range(linger_frames):
            frames.append(CameraFrame(lat=end[0], lon=end[1], zoom=prev_zoom, bearing=prev_bearing, seg_index=idx, seg_t=1.0))
    return frames


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
            line=dict(color="#00d1ff", width=3),
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
                marker=dict(size=local_marker + 2, color="#ff9f1c", line=dict(width=2, color="#2b2d42")),
                hovertext=marker_hover_text(df),
                hoverinfo="text",
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
                textfont=dict(color="#001219", size=max(9, int(local_marker * 0.5))),
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
                textfont=dict(color="#edf2f4", size=local_label_font),
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
                args=[[f"step-{i}"], {"frame": {"duration": 600, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                label=str(i + 1),
            )
            for i in range(0, len(df))
        ]
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1.08,
                    x=0.5,
                    xanchor="center",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 600, "redraw": False}, "fromcurrent": True, "transition": {"duration": 0}}],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0}, "mode": "immediate", "transition": {"duration": 0}}],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    y=0.02,
                    x=0.1,
                    len=0.8,
                    pad={"b": 10, "t": 50},
                    active=0,
                    steps=steps,
                )
            ],
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
    mapbox_style: str = "satellite",
    label_distance_km: float = 60.0,
    marker_size: int = 26,
    label_font_size: int = 13,
    progressive_route: bool = False,
    path_lat_override: Optional[List[float]] = None,
    path_lon_override: Optional[List[float]] = None,
    label_offset_km: float = 8.0,
    reached_upto: Optional[int] = None,
) -> go.Figure:
    path_lat, path_lon = build_path_segments(df)
    if path_lat_override is not None and path_lon_override is not None:
        path_lat, path_lon = path_lat_override, path_lon_override
    if progressive_route:
        path_lat, path_lon = [], []
    avg_lat = float(df["latitude"].mean()) if not df.empty else 0.0
    avg_lon = float(df["longitude"].mean()) if not df.empty else 0.0
    center_lat, center_lon = center or (avg_lat, avg_lon)

    fig = go.Figure()

    # Always add the path trace; it may start empty for progressive mode
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
        local_marker = max(10, int(marker_size))
        local_label_font = max(9, int(label_font_size))
        # Base shadow marker ring
        fig.add_trace(
            go.Scattermapbox(
                lat=df["latitude"],
                lon=df["longitude"],
                mode="markers",
                marker=dict(size=local_marker + 6, color="#023047", opacity=0.9),
                hoverinfo="skip",
            )
        )
        # Foreground marker with hover info
        fig.add_trace(
            go.Scattermapbox(
                lat=df["latitude"],
                lon=df["longitude"],
                mode="markers",
                marker=dict(size=local_marker, color="#ffb703", opacity=0.95),
                hovertext=marker_hover_text(df),
                hoverinfo="text",
            )
        )
        # Numbers overlay
        fig.add_trace(
            go.Scattermapbox(
                lat=df["latitude"],
                lon=df["longitude"],
                mode="text",
                text=marker_numbers(df),
                textposition="middle center",
                textfont=dict(color="#001219", size=max(9, int(local_marker * 0.5))),
                hoverinfo="skip",
            )
        )
        # Titles overlay; hide unreached if reached_upto provided
        full_titles = label_text(df)
        if reached_upto is None:
            active_upto = 0 if progressive_route else (len(df) - 1)
        else:
            active_upto = reached_upto
        titles = [t if i <= active_upto else "" for i, t in enumerate(full_titles)]
        fig.add_trace(
            go.Scattermapbox(
                lat=offset_latitudes(df["latitude"].tolist(), label_offset_km),
                lon=df["longitude"],
                mode="text",
                text=titles,
                textposition="top center",
                textfont=dict(color="#f7f9fb", size=local_label_font),
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
            style=mapbox_style,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom if zoom is not None else 1.4,
            bearing=bearing,
            pitch=pitch,
        ),
    )

    # Progressive frames: one per stop, route grows cumulatively
    if progressive_route and len(df) > 1:
        frames = []
        for upto in range(0, len(df)):
            part_df = df.iloc[: upto + 1].reset_index(drop=True)
            f_lat, f_lon = build_path_segments(part_df)
            titles = [t if i <= upto else "" for i, t in enumerate(label_text(df))]
            # traces: 0 path, 1 shadow markers, 2 markers, 3 numbers, 4 titles
            frames.append(
                go.Frame(
                    name=f"step-{upto}",
                    data=[
                        go.Scattermapbox(lat=f_lat, lon=f_lon),
                        go.Scattermapbox(text=titles),
                    ],
                    traces=[0, 4],
                )
            )
        fig.frames = frames
        steps = [
            dict(
                method="animate",
                args=[[f"step-{i}"], {"frame": {"duration": 600, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                label=str(i + 1),
            )
            for i in range(0, len(df))
        ]
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1.08,
                    x=0.5,
                    xanchor="center",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 600, "redraw": False}, "fromcurrent": True, "transition": {"duration": 0}}],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0}, "mode": "immediate", "transition": {"duration": 0}}],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    y=0.02,
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
    try:
        # Plotly/kaleido supports png, jpeg, webp, svg (we use raster formats)
        return fig.to_image(format=img_format, width=width, height=height)
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
            frame_defs = create_mapbox_camera_frames(df, args.fps, args.linger, args.zoom_boost)
            if not frame_defs:
                raise RuntimeError("Cannot build video: the CSV has no locations.")
            rendered: List = []
            for frame in frame_defs:
                p_lat, p_lon = build_partial_path_segments(df, frame.seg_index, frame.seg_t)
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
                    mapbox_style=args.mapbox_style,
                    label_distance_km=args.label_distance_km,
                    marker_size=max(12, int(args.marker_size * 0.9)),
                    label_font_size=max(10, int(args.label_font_size * 0.9)),
                    path_lat_override=p_lat,
                    path_lon_override=p_lon,
                    label_offset_km=args.label_offset_km,
                    reached_upto=(0 if frame.seg_index < 0 else frame.seg_index + (1 if frame.seg_t >= 1.0 else 0)),
                )
                image_bytes = figure_to_image(fig, img_format=args.frame_format, width=args.width, height=args.height)
                ext = ".jpg" if args.frame_format == "jpeg" else f".{args.frame_format}"
                rendered.append(iio.imread(image_bytes, extension=ext))
            return rendered

        frame_defs = create_globe_frames(df, args.fps, args.linger, args.zoom_boost)
        if not frame_defs:
            raise RuntimeError("Cannot build video: the CSV has no locations.")
        rendered: List = []
        for frame in frame_defs:
            p_lat, p_lon = build_partial_path_segments(df, frame.seg_index, frame.seg_t)
            fig = render_styled_geo(
                df,
                title=args.title,
                projection=args.projection,
                width=args.width,
                height=args.height,
                rotation=dict(lat=frame.lat, lon=frame.lon),
                scale=frame.scale,
                label_distance_km=args.label_distance_km,
                marker_size=max(12, int(args.marker_size * 0.9)),
                label_font_size=max(10, int(args.label_font_size * 0.9)),
                path_lat_override=p_lat,
                path_lon_override=p_lon,
                label_offset_km=args.label_offset_km,
                reached_upto=(0 if frame.seg_index < 0 else frame.seg_index + (1 if frame.seg_t >= 1.0 else 0)),
            )
            image_bytes = figure_to_image(fig, img_format=args.frame_format, width=args.width, height=args.height)
            ext = ".jpg" if args.frame_format == "jpeg" else f".{args.frame_format}"
            rendered.append(iio.imread(image_bytes, extension=ext))
        return rendered

    try:
        rendered_frames = render_mode_frames(preferred_mode)
    except MapboxTileUnavailable:
        if preferred_mode == "mapbox":
            print("⚠️  Mapbox tiles unavailable during video capture; falling back to the styled globe for MP4 output.")
            rendered_frames = render_mode_frames("styled")
        else:
            raise

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
        if chosen_fmt in codec_by_fmt:
            iio.imwrite(
                output_path,
                rendered_frames,
                fps=args.fps,
                codec=codec_by_fmt[chosen_fmt],
                bitrate=args.bitrate,
            )
        elif chosen_fmt == "gif":
            iio.imwrite(
                output_path,
                rendered_frames,
                fps=args.fps,
            )
        else:  # Fallback to mp4 if unknown
            iio.imwrite(
                output_path.with_suffix(".mp4"),
                rendered_frames,
                fps=args.fps,
                codec="libx264",
                bitrate=args.bitrate,
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
            mapbox_style=args.mapbox_style,
            label_distance_km=args.label_distance_km,
            marker_size=args.marker_size,
            label_font_size=args.label_font_size,
            progressive_route=(args.route_style == "progressive"),
            label_offset_km=args.label_offset_km,
        )
    else:
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
