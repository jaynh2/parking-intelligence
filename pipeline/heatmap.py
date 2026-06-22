"""
Phase 2 — Heatmap Generation.

The prototype dumped every raw (lat, lon) pair straight into folium's
HeatMap layer — on this dataset alone that produced a 19MB single HTML
file (one cell's output in the notebook), which does not scale and is a
poor experience to ship to a browser. Production fix: the heatmap LAYER
is rendered from a bounded random sample (settings.heatmap_max_points),
which is visually indistinguishable for density purposes, while the
HOTSPOT MARKERS (the part operators actually act on) are rendered from
the exact, fully-aggregated hotspot_summary — no precision lost where it
matters.
"""
from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd
from folium.plugins import HeatMap

from config.settings import Settings, get_settings
from pipeline.logging_config import get_logger

logger = get_logger(__name__)


def generate_heatmap(
    sample_points: pd.DataFrame,
    hotspot_summary: pd.DataFrame,
    output_path: Path,
    settings: Settings | None = None,
) -> Path:
    settings = settings or get_settings()

    if sample_points.empty:
        raise ValueError("Cannot render a heatmap with zero points")

    points = sample_points
    if len(points) > settings.heatmap_max_points:
        points = points.sample(n=settings.heatmap_max_points, random_state=settings.heatmap_sample_seed)

    center_lat = float(points["latitude"].mean())
    center_lon = float(points["longitude"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB dark_matter")

    HeatMap(
        points[["latitude", "longitude"]].values.tolist(),
        radius=15,
        blur=10,
        max_zoom=1,
        name="Violation Heatmap (sampled)",
    ).add_to(m)

    for _, row in hotspot_summary.iterrows():
        radius_size = max(5, min(40, row["total_violations"] / 5))
        popup_html = f"""
        <div style='width: 220px; font-family: sans-serif;'>
            <h4 style='margin-bottom: 5px; color: #d35400;'>{row['cluster_id']}</h4>
            <b>Violations:</b> <span style='color: red;'>{int(row['total_violations'])}</span><br>
            <b>Junction:</b> {row['junction_name']}<br>
            <b>Station:</b> {row['police_station']}<br>
            <b>Priority score:</b> {row.get('priority_score', 'n/a')}
        </div>
        """
        folium.CircleMarker(
            location=[row["center_latitude"], row["center_longitude"]],
            radius=radius_size,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.45,
            tooltip=f"{row['cluster_id']} — click for details",
            popup=folium.Popup(popup_html, max_width=260),
        ).add_to(m)

    folium.LayerControl().add_to(m)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logger.info("Phase 2 complete: heatmap with %d sampled points + %d hotspot markers -> %s",
                len(points), len(hotspot_summary), output_path)
    return output_path
