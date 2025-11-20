# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 15:57:20 2025

@author: ge92wex
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.graph_objects import Scatter

from PIL import Image
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

import re
from datetime import datetime


def extract_experiment_index(filename: str):
    match = re.search(r"experiment_(\d+)", filename)
    return int(match.group(1)) if match else float("inf")


def extract_operating_point(rel_path: str):
    for part in rel_path.split(os.sep):
        if part.startswith("operating_point_"):
            try:
                return int(part.split("_")[-1])
            except Exception:
                pass
    return float("inf")


def sort_key_multi(rel_path: str):
    return (extract_operating_point(rel_path),
            extract_experiment_index(os.path.basename(rel_path)))


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
    """Load YAML metadata for a given selected time-series file (if present)."""
    meta = os.path.splitext(sel)[0] + "_metadata.yaml"
    mpath = os.path.join(base, meta)
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


def main():
    # Page config & styling
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

    # Show P&ID if exists (now cached)
    img, size = load_pid_image("PID.png")
    if img is not None:
        st.subheader("P&ID of the mini-plant")

        w, h = size
        w = min(max(w, 900), 1800)
        h = min(max(h, 700), 1800)

        fig_pid = px.imshow(img)
        fig_pid.update_xaxes(
            visible=False, showticklabels=False, ticks="", showgrid=False, zeroline=False
        )
        fig_pid.update_yaxes(
            visible=False, showticklabels=False, ticks="", showgrid=False, zeroline=False
        )
        fig_pid.update_layout(
            xaxis=dict(visible=False, showticklabels=False, ticks="", showgrid=False, zeroline=False),
            yaxis=dict(visible=False, showticklabels=False, ticks="", showgrid=False, zeroline=False),
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
                    "toImage",
                    "autoScale2d",
                    "resetScale2d",
                    "lasso2d",
                    "select2d",
                ],
            },
            key="pid_plot_v2",
        )
    else:
        st.warning("P&ID not found at PID.png")

    # Scenario selection
    st.subheader("Select Feed Mixture / Scenario")
    mixture_to_folder = {
        "Water": "ScenarioA_SingleComponent_Water",
        "n-Butanol + Water": "ScenarioB_BinaryComponent_n-butanolwater",
        "OME": "ScenarioC_Reactive_OME",
    }
    mix = st.radio("Mixture:", list(mixture_to_folder.keys()), index=0)
    base = "ContinuousDistillationData"
    scenario_folder = os.path.join(base, mixture_to_folder[mix])
    if not os.path.isdir(scenario_folder):
        st.error(f"Folder not found: {scenario_folder}")
        return

    # Gather CSVs (cached)
    all_csv = gather_csvs(base, scenario_folder)
    if not all_csv:
        st.error("No CSVs found")
        return

    # Split into train/test & time/conc
    train_ts, train_conc, test_ts, test_conc = [], [], [], []
    for rel in all_csv:
        b = os.path.basename(rel).lower()
        if rel.startswith("train_normal_experiment_") or b.startswith("train_normal_experiment_"):
            (train_conc if "concentration" in b else train_ts).append(rel)
        elif rel.startswith("test_anormal_experiment_") or b.startswith("test_anormal_experiment_"):
            (test_conc if "concentration" in b else test_ts).append(rel)

    # Data type & experiment type
    data_type = st.radio("Select data type:", ["Time Series", "Concentration Data"])
    exp_type = st.radio("Experiment type:", ["Train (normal)", "Test (anormal)"])
    if data_type == "Time Series":
        options = train_ts if exp_type.startswith("Train") else test_ts
    else:
        options = train_conc if exp_type.startswith("Train") else test_conc

    if not options:
        st.warning(f"No {data_type.lower()} files")
        return

    sel = st.selectbox(f"Select a {data_type.lower()} file:", options)
    path_csv = os.path.join(base, sel)

    # Metadata only for time series (cached)
    if data_type == "Time Series":
        md = load_metadata(base, sel)
        st.subheader("Metadata:")
        if md is not None:
            st.json(md)
        else:
            st.info("No metadata for this file.")

    # Download button
    if os.path.isfile(path_csv):
        with open(path_csv, "rb") as f:
            st.download_button(
                "Download CSV", f, file_name=os.path.basename(path_csv)
            )
    else:
        st.error("CSV not found")
        return

    # Load into DataFrame (cached)
    try:
        df = load_csv(path_csv)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return

    # === Time-series branch ===
    if data_type == "Time Series":
        # Parse Time column
        if "Time" not in df.columns:
            st.error("Missing 'Time' column")
            return
        raw = df["Time"].astype(str)
        parsed = pd.to_datetime(raw, dayfirst=True, errors="coerce")
        bad = parsed.isna()
        if bad.any():
            prefix = datetime.today().strftime("%Y-%m-%d ")
            parsed2 = pd.to_datetime(prefix + raw[bad], errors="coerce")
            parsed.loc[bad] = parsed2
        df["Time"] = parsed
        df = df.dropna(subset=["Time"])

        # Cast Label if present
        if "Label (common/all)" in df.columns:
            df["Label (common/all)"] = df["Label (common/all)"].astype(str)

        # Show grid
        st.subheader("Data Preview:")
        for c in df.select_dtypes(include="bool"):
            df[c] = df[c].astype(str)

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(
            filterable=True,
            sortable=True,
            floatingFilter=True,
            resizable=True,
            minWidth=120,
        )
        if "Time" in df.columns:
            gb.configure_column("Time", pinned="left")

        gb.configure_pagination(paginationPageSize=15)
        grid = gb.build()

        AgGrid(
            df,
            gridOptions=grid,
            data_return_mode=DataReturnMode.AS_INPUT,
            update_mode=GridUpdateMode.NO_UPDATE,
            enable_enterprise_modules=True,
            height=400,
            fit_columns_on_grid_load=False,
        )

        # Time-series plotting
        df2 = df.sort_values("Time").copy()
        df2["Time"] = pd.to_datetime(df2["Time"], errors="coerce")

        dr = None
        if df2["Time"].notna().any():
            mn, mx = df2["Time"].min(), df2["Time"].max()
            if pd.notna(mn) and pd.notna(mx) and mn < mx:
                mn_py = mn.to_pydatetime()
                mx_py = mx.to_pydatetime()

                dr = st.slider(
                    "Time range:",
                    min_value=mn_py,
                    max_value=mx_py,
                    value=(mn_py, mx_py),
                    format="YYYY-MM-DD HH:mm:ss",
                )

        # Apply filter only if slider was created
        if dr is not None:
            df2 = df2[df2["Time"].between(dr[0], dr[1])]
            nums = [
                c
                for c in df2.columns
                if pd.api.types.is_numeric_dtype(df2[c]) and c != "Label (common/all)"
            ]
            default = ["T101"] if "T101" in nums else []
            cols = st.multiselect("Columns to plot:", nums, default=default)

            if cols:
                ptype = st.selectbox(
                    "Plot type:",
                    [
                        "Line Plot",
                        "Rolling Average",
                        "Heatmap",
                        "Autocorrelation",
                        "Seasonality Decomposition",
                    ],
                )

                ymi, yma = float(df2[cols].min().min()), float(df2[cols].max().max())
                yr = st.slider("Y-axis range:", ymi - 50, yma + 50, (ymi, yma))

                # ---------- (A) Line Plot ----------
                if ptype == "Line Plot":
                    fig_ts = px.line(df2, x="Time", y=cols, title="Time Series")
                    fig_ts.update_layout(
                        template="plotly_dark",
                        xaxis_title="Time",
                        yaxis_title="Value",
                    )
                    fig_ts.update_yaxes(range=yr)

                    # ---- Anomaly shading from "Label (common/hard fault)" == 1 ----
                    label_col = "Label (common/hard fault)"
                    if label_col in df2.columns:
                        df_filtered = df2[["Time", label_col]].copy()

                        is_one = (
                            pd.to_numeric(df_filtered[label_col], errors="coerce")
                            .fillna(0)
                            .astype(int)
                            == 1
                        )
                        df_filtered["is_anomaly"] = is_one
                        df_filtered["prev_anomaly"] = df_filtered["is_anomaly"].shift(
                            1, fill_value=False
                        )
                        df_filtered["start_of_block"] = (
                            df_filtered["is_anomaly"] & ~df_filtered["prev_anomaly"]
                        )
                        df_filtered["end_of_block"] = (
                            ~df_filtered["is_anomaly"] & df_filtered["prev_anomaly"]
                        )

                        start_times = df_filtered.loc[
                            df_filtered["start_of_block"], "Time"
                        ].tolist()
                        end_times = df_filtered.loc[
                            df_filtered["end_of_block"], "Time"
                        ].tolist()

                        if len(df_filtered) > 0 and df_filtered["is_anomaly"].iloc[-1]:
                            end_times.append(df_filtered["Time"].iloc[-1])

                        for s, e in zip(start_times, end_times):
                            fig_ts.add_vrect(
                                x0=s,
                                x1=e,
                                fillcolor="red",
                                opacity=0.3,
                                line_width=0,
                            )

                        anomaly_trace = Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            name="Anomaly",
                            line_color="red",
                            showlegend=True,
                            line=dict(width=1.5),
                        )
                        fig_ts.add_trace(anomaly_trace)
                    # ---------------------------------------------------------------

                    st.plotly_chart(fig_ts, use_container_width=True)

                # ---------- (B) Rolling Average ----------
                elif ptype == "Rolling Average":
                    window = st.number_input(
                        "Rolling window size",
                        min_value=2,
                        max_value=500,
                        value=10,
                        step=1,
                    )
                    df_roll = df2[["Time"] + cols].copy()
                    roll_cols = []
                    for c in cols:
                        rc = f"{c}_rolling"
                        df_roll[rc] = df_roll[c].rolling(window=window).mean()
                        roll_cols.append(rc)

                    fig_ra = px.line(
                        df_roll,
                        x="Time",
                        y=roll_cols,
                        title=f"Rolling Average (window={window})",
                    )
                    fig_ra.update_layout(
                        template="plotly_dark",
                        xaxis_title="Time",
                        yaxis_title="Value",
                    )
                    fig_ra.update_yaxes(range=yr)
                    st.plotly_chart(fig_ra, use_container_width=True)

                # ---------- (C) Heatmap ----------
                elif ptype == "Heatmap":
                    if len(cols) == 0:
                        st.warning("Select at least one column.")
                    else:
                        col0 = cols[0]
                        df_hm = df2[["Time", col0]].dropna().copy()
                        df_hm["hour"] = df_hm["Time"].dt.hour
                        df_hm["day"] = df_hm["Time"].dt.dayofweek
                        pivot = df_hm.pivot_table(
                            values=col0, index="day", columns="hour", aggfunc="mean"
                        )

                        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
                        sns.heatmap(pivot, ax=ax, cmap="coolwarm")
                        ax.set_xlabel("Hour of Day")
                        ax.set_ylabel("Day of Week (0=Mon)")
                        ax.set_title(f"Heatmap of {col0}")
                        st.pyplot(fig)

                # ---------- (E) Autocorrelation ----------
                elif ptype == "Autocorrelation":
                    if len(cols) == 0:
                        st.warning("Select at least one column.")
                    else:
                        from statsmodels.graphics.tsaplots import plot_acf

                        col0 = cols[0]
                        series = df2[col0].dropna()
                        if len(series) < 5:
                            st.warning("Not enough data for autocorrelation.")
                        else:
                            fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
                            plot_acf(series, lags=min(50, len(series) - 1), ax=ax)
                            ax.set_title(f"Autocorrelation of {col0}")
                            st.pyplot(fig)

                # ---------- (F) Seasonality Decomposition ----------
                elif ptype == "Seasonality Decomposition":
                    if len(cols) == 0:
                        st.warning("Select at least one column.")
                    else:
                        from statsmodels.tsa.seasonal import seasonal_decompose

                        col0 = cols[0]
                        series = df2[col0].dropna()
                        period = st.number_input(
                            "Seasonal period (samples)",
                            min_value=2,
                            max_value=2000,
                            value=50,
                            step=1,
                        )
                        if len(series) < 2 * period:
                            st.warning(
                                "Not enough data for seasonal decomposition with this period."
                            )
                        else:
                            res = seasonal_decompose(
                                series, model="additive", period=period
                            )
                            fig = res.plot()
                            fig.set_size_inches(8, 6)
                            st.pyplot(fig)

    # === Concentration branch ===
    else:
        st.subheader("Filtered Data Preview:")
        for c in df.select_dtypes(include="bool").columns:
            df[c] = df[c].astype(str)
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(
            filterable=True,
            sortable=True,
            floatingFilter=True,
            resizeable=True,
            min_width=120,
        )
        gb.configure_pagination(paginationPageSize=10)
        AgGrid(
            df,
            gridOptions=gb.build(),
            data_return_mode=DataReturnMode.AS_INPUT,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            enable_enterprise_modules=True,
        )

        if "PackingHeight" not in df.columns:
            st.info("No 'PackingHeight' column available for plotting.")
            return

        # Find all MassFraction columns
        mass_cols = [c for c in df.columns if c.startswith("MassFraction")]
        if not mass_cols:
            st.warning("No 'MassFraction...' columns found in this file.")
            return

        st.write("Plotting Packing Height vs. Mass Fraction")
        mass_col = st.selectbox("Select which MassFraction to plot:", mass_cols)

        # slider bounds from the data
        mf_min = float(df[mass_col].min())
        mf_max = float(df[mass_col].max())
        mf_range = st.slider(
            f"Select {mass_col} range:",
            min_value=mf_min,
            max_value=mf_max,
            value=(mf_min, mf_max),
            step=(mf_max - mf_min) / 100 if mf_max > mf_min else 0.001,
        )

        df_filt = df[
            (df[mass_col] >= mf_range[0]) & (df[mass_col] <= mf_range[1])
        ]

        fig_sc = px.scatter(
            df_filt,
            x=mass_col,
            y="PackingHeight",
            title=f"PackingHeight vs {mass_col}",
        )
        fig_sc.update_layout(
            xaxis_title=mass_col,
            yaxis_title="PackingHeight (m)",
            template="plotly_dark",
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # === Sensor Features lookup ===
    st.write("---")
    st.subheader("Search Sensor Features")

    feat = load_features(scenario_folder)
    if feat.empty:
        st.info("No Features file found.")
    else:
        q = st.text_input("SensorActuatorName:")
        if st.button("Search Features"):
            if {"SensorActuatorName", "Description"}.issubset(feat.columns):
                res = feat[
                    feat["SensorActuatorName"].str.upper()
                    == q.strip().upper()
                ]
            else:
                res = pd.DataFrame()

            if not res.empty:
                for _, r in res.iterrows():
                    st.markdown(
                        f"**{r['SensorActuatorName']}**: {r['Description']}"
                    )
            else:
                st.warning("No matches found.")


if __name__ == "__main__":
    main()


