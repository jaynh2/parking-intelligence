"""
Production dashboard. Unlike the original prototype, this file contains
ZERO hardcoded data and ZERO model/clustering logic — it is a pure
presentation layer that calls the inference API (inference/api.py) over
HTTP and renders whatever comes back. There is no retrain button and no
code path that touches pipeline/ at all: training happens out-of-band
(scheduled job / separate container), and this process only ever reads
the latest result through the API's cache.

Run:
    streamlit run dashboard/app.py
Requires PCI_API_BASE_URL (or the default http://localhost:8000) to be reachable.
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

API_BASE_URL = os.environ.get("PCI_API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("PCI_API_KEY", "")          # empty → dev-mode, no auth header sent
CACHE_TTL = int(os.environ.get("PCI_DASHBOARD_CACHE_TTL", "60"))

_AUTH_HEADERS: dict[str, str] = ({"X-API-Key": API_KEY} if API_KEY else {})

st.set_page_config(page_title="Bengaluru Traffic Control AI", layout="wide", page_icon="🚨")


# ---------------------------------------------------------------------- #
# Low-level retry wrappers — transient connection errors retried 3× with
# exponential backoff (1s → 2s → 4s). Timeouts and HTTP 4xx/5xx are NOT
# retried — those indicate a real problem, not a blip.
# ---------------------------------------------------------------------- #
@retry(
    retry=retry_if_exception_type(requests.ConnectionError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _get(url: str, **kwargs) -> requests.Response:
    return requests.get(url, **kwargs)  # nosec B113 — timeout always passed via **kwargs by callers


@retry(
    retry=retry_if_exception_type(requests.ConnectionError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _post(url: str, **kwargs) -> requests.Response:
    return requests.post(url, **kwargs)  # nosec B113 — timeout always passed via **kwargs by callers


# ---------------------------------------------------------------------- #
# API client helpers — every call is cached and every call can fail
# independently; a down endpoint degrades its section, not the whole page.
# ---------------------------------------------------------------------- #
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_json(path: str, params: dict | None = None) -> dict | list | None:
    try:
        resp = _get(f"{API_BASE_URL}{path}", params=params, headers=_AUTH_HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        return {"__error__": str(exc)}


def fetch_health() -> dict:
    try:
        resp = _get(f"{API_BASE_URL}/health", timeout=5)  # /health is always public
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        return {"status": "unreachable", "artifacts_loaded": False, "error": str(exc)}


def is_error(payload) -> bool:
    return isinstance(payload, dict) and "__error__" in payload


# ---------------------------------------------------------------------- #
# Header + connectivity banner
# ---------------------------------------------------------------------- #
st.title("🚨 AI-Driven Parking Intelligence & Congestion Command Center")
st.markdown("Real-time enforcement tasking, predictive analytics, and spatial risk optimization.")

health = fetch_health()
if health.get("status") == "ok":
    st.caption(
        f"🟢 Connected to inference API — model `{health.get('model_version')}`, "
        f"artifacts generated {health.get('artifact_generated_at')}"
    )
elif health.get("status") == "degraded":
    st.warning("🟡 API is up but no trained artifacts are available yet. Run the training pipeline first.")
    st.stop()
else:
    st.error(
        f"🔴 Cannot reach the inference API at `{API_BASE_URL}`. "
        f"Start it with `uvicorn inference.api:app` or check PCI_API_BASE_URL. ({health.get('error', '')})"
    )
    st.stop()

st.write("---")

# ---------------------------------------------------------------------- #
# KPI cards
# ---------------------------------------------------------------------- #
kpis = fetch_json("/api/v1/kpis")
col1, col2, col3, col4 = st.columns(4)
if is_error(kpis):
    st.error(f"KPI service unavailable: {kpis['__error__']}")
else:
    with col1:
        st.metric(label="Total Active Hotspots Identified", value=f"{kpis['total_active_hotspots']:,} Zones")
    with col2:
        st.metric(
            label="Critical Dispatch Threshold",
            value=f"{kpis['top_priority_score']:,.1f} PTS",
            delta=kpis["top_priority_junction"],
            delta_color="inverse",
        )
    with col3:
        st.metric(label="AI Predictive Error Window", value=f"± {kpis['model_mae']:.2f} Vehicles/hr")
    with col4:
        st.metric(label="Primary Congestion Offense", value=kpis["primary_offense"].title(), delta="High Severity")

st.write("---")

# ---------------------------------------------------------------------- #
# Leaderboard + dispatch directives | Feature importance
# ---------------------------------------------------------------------- #
left_col, right_col = st.columns([5, 4])

leaderboard_resp = fetch_json("/api/v1/leaderboard", params={"limit": 25})

with left_col:
    st.subheader("📋 Priority Enforcement Leaderboard")
    if is_error(leaderboard_resp):
        st.error(f"Leaderboard service unavailable: {leaderboard_resp['__error__']}")
    else:
        hotspots = leaderboard_resp["hotspots"]
        st.caption(f"Showing top {len(hotspots)} of {leaderboard_resp['total_hotspots']:,} tracked hotspots")
        board_df = pd.DataFrame(hotspots)
        st.dataframe(
            board_df[["rank", "junction_name", "police_station", "total_violations", "priority_score", "status"]]
            .rename(columns={
                "rank": "Rank", "junction_name": "Junction", "police_station": "Station",
                "total_violations": "Violations", "priority_score": "Priority Score", "status": "Status",
            }),
            use_container_width=True, hide_index=True,
        )

        st.subheader("💡 Automated Dispatch Directives")
        for row in hotspots[:8]:
            label = f"**Rank {row['rank']}: {row['status'].replace('_', ' ')}** — {row['recommendation']}"
            if row["status"] == "URGENT_DISPATCH":
                st.error(label)
            elif row["status"] == "HIGH_PRIORITY":
                st.warning(label)
            else:
                st.info(label)

with right_col:
    st.subheader("📈 AI Temporal Insights & Features")
    importance = fetch_json("/api/v1/feature-importance")
    if is_error(importance):
        st.error(f"Feature importance unavailable: {importance['__error__']}")
    else:
        fi_df = pd.DataFrame(importance).sort_values("importance", ascending=True)
        fig = px.bar(
            fi_df, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Viridis",
            title="What Drives Congestion Predictability?",
        )
        fig.update_layout(showlegend=False, height=280, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔮 Forecast a Specific Hotspot")
    if not is_error(leaderboard_resp) and leaderboard_resp["hotspots"]:
        options = {f"{h['junction_name']} ({h['cluster_id']})": h["cluster_id"] for h in leaderboard_resp["hotspots"]}
        choice = st.selectbox("Hotspot", list(options.keys()))
        dow = st.selectbox("Day of week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=4)
        hour = st.slider("Hour of day", 0, 23, 18)
        if st.button("Predict violation volume"):
            payload = {"cluster_id": options[choice], "day_of_week": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(dow), "hour": hour}
            try:
                resp = _post(f"{API_BASE_URL}/api/v1/predict", json=payload, headers=_AUTH_HEADERS, timeout=10)
                if resp.status_code == 200:
                    pred = resp.json()
                    st.success(f"Predicted **{pred['predicted_violations']:.1f}** violations (± {pred['model_mae']:.1f} MAE)")
                else:
                    st.error(resp.json().get("detail", f"Request failed ({resp.status_code})"))
            except requests.RequestException as exc:
                st.error(f"Prediction request failed: {exc}")

st.write("---")

# ---------------------------------------------------------------------- #
# Embedded geospatial heatmap
# ---------------------------------------------------------------------- #
st.subheader("🗺️ Live Geospatial Heatmap & Active Hotspot Centroids")
st.markdown("Toggle map layers to view real-world coordinate clusters and violation density fields.")
try:
    map_resp = _get(f"{API_BASE_URL}/api/v1/map", headers=_AUTH_HEADERS, timeout=15)
    if map_resp.status_code == 200:
        components.html(map_resp.text, height=600, scrolling=True)
    else:
        st.info(f"Map not available yet ({map_resp.status_code}).")
except requests.RequestException as exc:
    st.info(f"Could not load the map from the API: {exc}")

st.caption("This dashboard performs no computation of its own — every number above is read live from the inference API's cached artifacts.")
