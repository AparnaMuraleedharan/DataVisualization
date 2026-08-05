# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Technical University of Munich
# SPDX-License-Identifier: MIT
#
# Visualization app for the continuous distillation dataset.
# Dataset: https://doi.org/10.5281/zenodo.17628963
#
# Author: Aparna Muraleedharan

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
import yaml
from PIL import Image
from plotly.graph_objects import Scatter
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, GridUpdateMode

# Data root. Defaults to a folder named ContinuousDistillationData next to this
# file; override with the DATA_ROOT environment variable.
BASE = os.environ.get("DATA_ROOT", "ContinuousDistillationData")

# Arbitrary anchor date. The CSV files carry clock time only (HH:MM:SS, no
# date), so a date is attached purely to make the values sortable and plottable.
ANCHOR_DATE = pd.Timestamp("2000-01-01")


def extract_experiment_index(filename: str):
    match = re.search(r"experiment_(\d+)", filename)
    return int(match.group(1)) if match else float("inf")


def extract_operating_point(rel_path: str):
    for part in rel_path.split(os.sep):
        if part.startswith("operating_point_"):
            try:
                return int(part.split("_")[-1])
            except ValueError:
                pass
    return float("inf")


def sort_key_multi(rel_path: str):
    return (extract_operating_point(rel_path),
            extract_experiment_index(os.path.basename(rel_path)))


def parse_clock_time(raw: pd.Series) -> pd.Series:
    """Turn a column of clock times (HH:MM:SS, no date) into timestamps.

    Rows in a file run consecutively, so a reading that is earlier than the one
    before it means the run has crossed midnight. Each such wrap adds one day,
    which keeps overnight runs in the correct order. Falls back to a plain
    datetime parse if the column already contains dates.
    """
    raw = raw.astype(str).str.strip()

    as_td = pd.to_timedelta(raw, errors="coerce")
    if as_td.notna().mean() >= 0.5:
        wraps = (as_td.diff() < pd.Timedelta(0)).fillna(False).cumsum()
        return ANCHOR_DATE + as_td + pd.to_timedelta(wraps, unit="D")

    return pd.to_datetime(raw, dayfirst=True, errors="coerce")


# ---------- CACHED HELPERS (for speed) ----------

@st.cache_data
def gather_csvs(base: str, scenario_folder: str):
    """Return sorted list of all CSVs for this scenario."""
    all_csv = []
    for root, _, files in os.walk(scenario_folder):
        for f in files:
            if f.endswith(".csv"):
                rel = os.path.join(os.path.relpath(root, base), f)
                all_csv.append(rel)
    all_csv.sort(key=sort_key_multi)
    return all_csv


@st.cache_data
def load_csv(path_csv: str):
    """Load a CSV into a DataFrame."""
    return pd.read_csv(path_csv)


@st.cache_data
def load_metadata(base: str, sel: str):
    """Load YAML metadata for a given selected time-series file (if present).

    Accepts either <name>_metadata.yaml or <name>.yaml.
    """
    stem = os.path.splitext(sel)[0]
    for candidate in (stem + "_metadata.yaml", stem + ".yaml"):
        mpath = os.path.join(base, candidate)
        if os.path.isfile(mpath):
            with open(mpath, "r") as f:
                return yaml.safe_load(f)
    return None


@st.cache_data
def load_features(scenario_folder: str):
    """Load Features CSV (if any) for the selected scenario."""
    feat_files = [
        fn for fn in os.listdir(scenario_folder)
        if fn.startswith("Features") and fn.lower().endswith(".csv")
    ]
    if not feat_files:
        return pd.DataFrame()
    feat_path = os.path.join(scenario_folder, feat_files[0])
    return pd.read_csv(feat_path, dtype=str)


@st.cache_data
def load_pid_image(path: str = "PID.png"):
    """Load the P&ID image (if it exists)."""
    if os.path.isfile(path):
        img = Image.open(path).convert("RGBA")
        return img, img.size
    return None, (None, None)


# ------------------------------------------------


def render_pid():
    img, size = load_pid_image("PID.png")
    if img is None:
        st.warning("P&ID not found at PID.png")
        return

    st.subheader("P&ID of the mini-plant")

    w, h = size
    w = min(max(w, 900), 1800)
    h = min(max(h, 700), 1800)

    fig_pid = px.imshow(img)
    axis_off = dict(visible=False, showticklabels=False, ticks="",
                    showgrid=False, zeroline=False)
    fig_pid.update_xaxes(**axis_off)
    fig_pid.update_yaxes(**axis_off)
    fig_pid.update_layout(
        xaxis=axis_off,
        yaxis=axis_off,
        margin=dict(l=0, r=0, t=0, b=0),
        width=w,
        height=h,
        autosize=True,
        dragmode="pan",
        uirevision="pid-v1",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig_pid,
        use_container_width=True,
        theme=None,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "scrollZoom": True,
            "doubleClick": "reset",
            "modeBarButtonsToRemove": [
                "toImage", "autoScale2d", "resetScale2d", "lasso2d", "select2d",
            ],
        },
        key="pid_plot_v2",
    )


def show_grid(df: pd.DataFrame, page_size: int = 15, pin_time: bool = False):
    """Render a DataFrame in an AG Grid (Community modules only)."""
    df = df.copy()
    for c in df.select_dtypes(include="bool").columns:
        df[c] = df[c].astype(str)

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        filterable=True,
        sortable=True,
        floatingFilter=True,
        resizable=True,
        minWidth=120,
    )
    if pin_time and "Time" in df.columns:
        gb.configure_column("Time", pinned="left")
    gb.configure_pagination(paginationPageSize=page_size)

    AgGrid(
        df,
        gridOptions=gb.build(),
        data_return_mode=DataReturnMode.AS_INPUT,
        update_mode=GridUpdateMode.NO_UPDATE,
        enable_enterprise_modules=False,
        height=400,
        fit_columns_on_grid_load=False,
    )


def add_anomaly_shading(fig, df2: pd.DataFrame, label_col: str):
    """Shade the periods in which the selected label column is non-zero."""
    if label_col == "None" or label_col not in df2.columns:
        return

    marks = df2[["Time", label_col]].copy()
    flag = pd.to_numeric(marks[label_col], errors="coerce").fillna(0) != 0
    marks["is_anomaly"] = flag
    marks["prev_anomaly"] = marks["is_anomaly"].shift(1, fill_value=False)
    marks["start_of_block"] = marks["is_anomaly"] & ~marks["prev_anomaly"]
    marks["end_of_block"] = ~marks["is_anomaly"] & marks["prev_anomaly"]

    start_times = marks.loc[marks["start_of_block"], "Time"].tolist()
    end_times = marks.loc[marks["end_of_block"], "Time"].tolist()

    # If the run ends while still anomalous, extend to the last timestamp.
    if len(marks) > 0 and marks["is_anomaly"].iloc[-1]:
        end_times.append(marks["Time"].iloc[-1])

    if not start_times:
        return

    for s, e in zip(start_times, end_times):
        fig.add_vrect(x0=s, x1=e, fillcolor="red", opacity=0.3, line_width=0)

    fig.add_trace(Scatter(
        x=[None], y=[None], mode="lines", name="Anomaly",
        line_color="red", showlegend=True, line=dict(width=1.5),
    ))


def main():
    st.set_page_config(page_title="Data Visualizer", page_icon="📊", layout="wide")
    st.markdown(
        """
        <style>
            .title {font-size:48px; color:#ff6347; text-align:center;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="title">Continuous Distillation Data Visualizer</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "This app allows you to visualize data from your CSV files. "
        "You can also filter by date and value range, and detect anomalies within the data."
    )

    render_pid()

    # Scenario selection
    st.subheader("Select Feed Mixture / Scenario")
    mixture_to_folder = {
        "Water": "ScenarioA_SingleComponent_Water",
        "n-Butanol + Water": "ScenarioB_BinaryComponent_n-butanolwater",
        "OME": "ScenarioC_Reactive_OME",
    }
    mix = st.radio("Mixture:", list(mixture_to_folder.keys()), index=0)
    scenario_folder = os.path.join(BASE, mixture_to_folder[mix])
    if not os.path.isdir(scenario_folder):
        st.error(
            f"Folder not found: {scenario_folder}. Set the DATA_ROOT environment "
            "variable to the location of the unzipped dataset."
        )
        return

    all_csv = gather_csvs(BASE, scenario_folder)
    if not all_csv:
        st.error("No CSVs found")
        return

    # Split into normal/anomalous and time series/concentration.
    normal_ts, normal_conc, anormal_ts, anormal_conc = [], [], [], []
    for rel in all_csv:
        b = os.path.basename(rel).lower()
        if "anormal_experiment_" in b:
            (anormal_conc if "concentration" in b else anormal_ts).append(rel)
        elif "normal_experiment_" in b:
            (normal_conc if "concentration" in b else normal_ts).append(rel)

    data_type = st.radio("Select data type:", ["Time Series", "Concentration Data"])
    exp_type = st.radio("Experiment type:", ["Normal", "Anomalous"])
    if data_type == "Time Series":
        options = normal_ts if exp_type == "Normal" else anormal_ts
    else:
        options = normal_conc if exp_type == "Normal" else anormal_conc

    if not options:
        st.warning(f"No {data_type.lower()} files")
        return

    sel = st.selectbox(f"Select a {data_type.lower()} file:", options)
    path_csv = os.path.join(BASE, sel)

    if data_type == "Time Series":
        md = load_metadata(BASE, sel)
        st.subheader("Metadata:")
        if md is not None:
            st.json(md)
        else:
            st.info("No metadata for this file.")

    if os.path.isfile(path_csv):
        with open(path_csv, "rb") as f:
            st.download_button("Download CSV", f,
                               file_name=os.path.basename(path_csv))
    else:
        st.error("CSV not found")
        return

    try:
        df = load_csv(path_csv)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return

    # === Time-series branch ===
    if data_type == "Time Series":
        if "Time" not in df.columns:
            st.error("Missing 'Time' column")
            return

        df = df.copy()
        df["Time"] = parse_clock_time(df["Time"])
        df = df.dropna(subset=["Time"])
        if df.empty:
            st.error("No valid timestamps in this file.")
            return

        if "Label (common/all)" in df.columns:
            df["Label (common/all)"] = df["Label (common/all)"].astype(str)

        st.subheader("Data Preview:")
        show_grid(df, page_size=15, pin_time=True)
        st.caption(
            "Times are clock times without a date. Runs that cross midnight are "
            "unwrapped onto consecutive days so that they plot in order."
        )

        df2 = df.sort_values("Time").copy()

        # Optional time-range filter. Only offered when the run spans a range;
        # plotting stays available either way.
        mn, mx = df2["Time"].min(), df2["Time"].max()
        if pd.notna(mn) and pd.notna(mx) and mn < mx:
            dr = st.slider(
                "Time range:",
                min_value=mn.to_pydatetime(),
                max_value=mx.to_pydatetime(),
                value=(mn.to_pydatetime(), mx.to_pydatetime()),
                format="HH:mm:ss",
            )
            df2 = df2[df2["Time"].between(dr[0], dr[1])]

        if df2.empty:
            st.warning("No rows in the selected time range.")
            return

        label_cols = [c for c in df2.columns if c.startswith("Label (")]
        nums = [
            c for c in df2.columns
            if pd.api.types.is_numeric_dtype(df2[c]) and c not in label_cols
        ]
        if not nums:
            st.warning("No numeric columns to plot in this file.")
            return

        default = ["T101"] if "T101" in nums else nums[:1]
        cols = st.multiselect("Columns to plot:", nums, default=default)
        if not cols:
            return

        ptype = st.selectbox(
            "Plot type:",
            ["Line Plot", "Rolling Average", "Heatmap",
             "Autocorrelation", "Seasonality Decomposition"],
        )

        ymi = float(df2[cols].min().min())
        yma = float(df2[cols].max().max())
        margin = 0.05 * (yma - ymi) if yma > ymi else max(abs(yma) * 0.05, 1e-6)
        yr = st.slider("Y-axis range:", ymi - margin, yma + margin, (ymi, yma))

        # ---------- (A) Line Plot ----------
        if ptype == "Line Plot":
            fig_ts = px.line(df2, x="Time", y=cols, title="Time Series")
            fig_ts.update_layout(template="plotly_dark",
                                 xaxis_title="Time", yaxis_title="Value")
            fig_ts.update_yaxes(range=yr)

            if label_cols:
                default_label = ("Label (common/hard fault)"
                                 if "Label (common/hard fault)" in label_cols
                                 else label_cols[0])
                shade_col = st.selectbox(
                    "Shade periods marked by:",
                    ["None"] + label_cols,
                    index=(label_cols.index(default_label) + 1),
                )
                add_anomaly_shading(fig_ts, df2, shade_col)

            st.plotly_chart(fig_ts, use_container_width=True)

        # ---------- (B) Rolling Average ----------
        elif ptype == "Rolling Average":
            window = st.number_input("Rolling window size", min_value=2,
                                     max_value=500, value=10, step=1)
            df_roll = df2[["Time"] + cols].copy()
            roll_cols = []
            for c in cols:
                rc = f"{c}_rolling"
                df_roll[rc] = df_roll[c].rolling(window=window).mean()
                roll_cols.append(rc)

            fig_ra = px.line(df_roll, x="Time", y=roll_cols,
                             title=f"Rolling Average (window={window})")
            fig_ra.update_layout(template="plotly_dark",
                                 xaxis_title="Time", yaxis_title="Value")
            fig_ra.update_yaxes(range=yr)
            st.plotly_chart(fig_ra, use_container_width=True)

        # ---------- (C) Heatmap ----------
        elif ptype == "Heatmap":
            st.caption(
                "Each selected sensor is averaged within equally spaced time "
                "bins. Normalizing puts sensors with different units on a "
                "common scale so their patterns can be compared."
            )
            n_bins = st.number_input("Number of time bins", min_value=10,
                                     max_value=500, value=100, step=10)
            normalize = st.checkbox("Normalize each sensor (z-score)", value=True)

            df_hm = df2[["Time"] + cols].dropna(subset=["Time"]).copy()
            df_hm["bin"] = pd.cut(df_hm["Time"].astype("int64"),
                                  bins=int(n_bins), labels=False)
            pivot = df_hm.groupby("bin")[cols].mean().T

            if normalize:
                spread = pivot.std(axis=1).replace(0, 1)
                pivot = pivot.sub(pivot.mean(axis=1), axis=0).div(spread, axis=0)

            bin_starts = df_hm.groupby("bin")["Time"].min()
            labels = [bin_starts.get(b, pd.NaT) for b in pivot.columns]
            labels = [t.strftime("%H:%M") if pd.notna(t) else "" for t in labels]
            step = max(1, len(labels) // 12)
            xticks = [lab if i % step == 0 else "" for i, lab in enumerate(labels)]

            fig, ax = plt.subplots(figsize=(10, max(2.5, 0.4 * len(cols))), dpi=100)
            sns.heatmap(pivot, ax=ax, cmap="coolwarm", xticklabels=xticks)
            ax.set_xlabel("Time")
            ax.set_ylabel("Sensor")
            ax.set_title("Sensor values over time"
                         + (" (normalized)" if normalize else ""))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ---------- (D) Autocorrelation ----------
        elif ptype == "Autocorrelation":
            from statsmodels.graphics.tsaplots import plot_acf

            col0 = cols[0]
            series = df2[col0].dropna()
            if len(series) < 5:
                st.warning("Not enough data for autocorrelation.")
            else:
                fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
                plot_acf(series, lags=min(50, len(series) - 1), ax=ax)
                ax.set_title(f"Autocorrelation of {col0}")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # ---------- (E) Seasonality Decomposition ----------
        elif ptype == "Seasonality Decomposition":
            from statsmodels.tsa.seasonal import seasonal_decompose

            col0 = cols[0]
            series = df2[col0].dropna()
            period = st.number_input("Seasonal period (samples)", min_value=2,
                                     max_value=2000, value=50, step=1)
            if len(series) < 2 * period:
                st.warning(
                    "Not enough data for seasonal decomposition with this period."
                )
            else:
                res = seasonal_decompose(series, model="additive",
                                         period=int(period))
                fig = res.plot()
                fig.set_size_inches(8, 6)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # === Concentration branch ===
    else:
        st.subheader("Filtered Data Preview:")
        show_grid(df, page_size=10)

        if "PackingHeight" not in df.columns:
            st.info("No 'PackingHeight' column available for plotting.")
            return

        mass_cols = [c for c in df.columns if c.startswith("MassFraction")]
        if not mass_cols:
            st.warning("No 'MassFraction...' columns found in this file.")
            return

        st.write("Plotting Packing Height vs. Mass Fraction")
        mass_col = st.selectbox("Select which MassFraction to plot:", mass_cols)

        mf_min = float(df[mass_col].min())
        mf_max = float(df[mass_col].max())
        mf_range = st.slider(
            f"Select {mass_col} range:",
            min_value=mf_min,
            max_value=mf_max,
            value=(mf_min, mf_max),
            step=(mf_max - mf_min) / 100 if mf_max > mf_min else 0.001,
        )

        df_filt = df[(df[mass_col] >= mf_range[0]) & (df[mass_col] <= mf_range[1])]

        fig_sc = px.scatter(df_filt, x=mass_col, y="PackingHeight",
                            title=f"PackingHeight vs {mass_col}")
        fig_sc.update_layout(xaxis_title=mass_col,
                             yaxis_title="PackingHeight (m)",
                             template="plotly_dark")
        st.plotly_chart(fig_sc, use_container_width=True)

    # === Sensor Features lookup ===
    st.write("---")
    st.subheader("Search Sensor Features")

    feat = load_features(scenario_folder)
    if feat.empty:
        st.info("No Features file found.")
        return

    name_col = next(
        (c for c in ("SensorActuatorName", "SensorName") if c in feat.columns),
        None,
    )
    if name_col is None or "Description" not in feat.columns:
        st.warning("Features file does not have the expected columns.")
        return

    q = st.text_input("SensorActuatorName:")
    if st.button("Search Features"):
        res = feat[feat[name_col].str.upper() == q.strip().upper()]
        if not res.empty:
            for _, r in res.iterrows():
                st.markdown(f"**{r[name_col]}**: {r['Description']}")
        else:
            st.warning("No matches found.")


if __name__ == "__main__":
    main()
