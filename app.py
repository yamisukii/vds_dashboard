# app.py
import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Load data (parquet preferred)
# ----------------------------
def load_dashboard_df():
    try:
        df = pd.read_parquet("dashboard_df.parquet")
    except Exception:
        df = pd.read_csv("dashboard_df.csv")

    df["datetime"] = pd.to_datetime(df["datetime"])
    # ensure required columns exist
    required = {"datetime", "y_true", "y_pred", "resid", "abs_err", "hour", "month", "cbwd"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dashboard_df: {missing}")

    # stable row id for selection/brushing
    df = df.sort_values("datetime").reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df

def load_importance():
    try:
        imp = pd.read_csv("feature_importance.csv")
        if not {"feature", "importance"}.issubset(set(imp.columns)):
            raise ValueError("feature_importance.csv must have columns: feature, importance")
        imp = imp.sort_values("importance", ascending=False).head(15).sort_values("importance")
        return imp
    except Exception:
        return pd.DataFrame({"feature": ["(no feature_importance.csv found)"], "importance": [0.0]})

df = load_dashboard_df()
imp = load_importance()

cbwd_options = sorted(df["cbwd"].dropna().unique().tolist())

# ----------------------------
# Helpers
# ----------------------------
def apply_global_filters(base_df, months_range, cbwd_list):
    out = base_df
    m0, m1 = months_range
    out = out[(out["month"] >= m0) & (out["month"] <= m1)]
    if cbwd_list:
        out = out[out["cbwd"].isin(cbwd_list)]
    return out

def apply_time_brush(base_df, relayoutData):
    if not relayoutData:
        return base_df

    x0 = relayoutData.get("xaxis.range[0]")
    x1 = relayoutData.get("xaxis.range[1]")
    if x0 and x1:
        x0 = pd.to_datetime(x0)
        x1 = pd.to_datetime(x1)
        return base_df[(base_df["datetime"] >= x0) & (base_df["datetime"] <= x1)]

    # if user clicked autoscale
    if relayoutData.get("xaxis.autorange"):
        return base_df

    return base_df

def apply_scatter_selection(base_df, selectedData):
    if not selectedData or "points" not in selectedData:
        return base_df
    ids = []
    for p in selectedData["points"]:
        cd = p.get("customdata")
        if cd is None:
            continue
        # customdata is a list: [row_id]
        ids.append(cd[0] if isinstance(cd, (list, tuple)) else cd)
    if not ids:
        return base_df
    return base_df[base_df["row_id"].isin(ids)]

def calc_metrics(d):
    if len(d) == 0:
        return {"n": 0, "r2": np.nan, "rmse": np.nan, "mae": np.nan}
    y = d["y_true"].values
    p = d["y_pred"].values
    resid = y - p
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean(resid**2))
    mae = np.mean(np.abs(resid))
    return {"n": len(d), "r2": r2, "rmse": rmse, "mae": mae}

# ----------------------------
# App
# ----------------------------
app = Dash(__name__)
app.title = "PM2.5 Dashboard"

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "Arial"},
    children=[
        html.H2("PM2.5 Random Forest — Dashboard"),

        # Controls
        html.Div(
            style={"display": "flex", "gap": "16px", "alignItems": "center", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={"minWidth": "340px"},
                    children=[
                        html.Div("Wind direction (cbwd)"),
                        dcc.Dropdown(
                            id="cbwd_dropdown",
                            options=[{"label": c, "value": c} for c in cbwd_options],
                            value=cbwd_options,  # all by default
                            multi=True,
                            clearable=False,
                        ),
                    ],
                ),

                html.Div(
                    style={"flex": "1", "minWidth": "420px"},
                    children=[
                        html.Div("Month range"),
                        dcc.RangeSlider(
                            id="month_slider",
                            min=1, max=12, step=1,
                            value=[1, 12],
                            marks={i: str(i) for i in range(1, 13)},
                        ),
                    ],
                ),

                html.Button("Reset", id="reset_btn", n_clicks=0),
                dcc.Store(id="reset_key", data=0),
            ],
        ),

        html.Hr(),

        # KPIs
        html.Div(
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            children=[
                html.Div(id="kpi_n", style={"padding": "10px", "border": "1px solid #ddd", "borderRadius": "10px", "minWidth": "170px"}),
                html.Div(id="kpi_r2", style={"padding": "10px", "border": "1px solid #ddd", "borderRadius": "10px", "minWidth": "170px"}),
                html.Div(id="kpi_rmse", style={"padding": "10px", "border": "1px solid #ddd", "borderRadius": "10px", "minWidth": "170px"}),
                html.Div(id="kpi_mae", style={"padding": "10px", "border": "1px solid #ddd", "borderRadius": "10px", "minWidth": "170px"}),
            ],
        ),

        html.Hr(),

        # --- TOP ROW: Residual view (as requested) ---
        html.Div(
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={"flex": "1", "minWidth": "620px"},
                    children=[dcc.Graph(id="resid_scatter", style={"height": "320px"}, config={"scrollZoom": True})],
                ),
                html.Div(
                    style={"flex": "1", "minWidth": "520px"},
                    children=[dcc.Graph(id="resid_hist", style={"height": "320px"}, config={"scrollZoom": True})],
                ),
            ],
        ),

        # Time series (zoom/brush filters all other views)
        dcc.Graph(id="ts_graph", style={"height": "320px"}, config={"scrollZoom": True}),

        # Pred vs actual + cbwd boxplot
        html.Div(
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={"flex": "1", "minWidth": "620px"},
                    children=[dcc.Graph(id="scatter_graph", style={"height": "320px"}, config={"scrollZoom": True})],
                ),
                html.Div(
                    style={"flex": "1", "minWidth": "520px"},
                    children=[dcc.Graph(id="cbwd_box", style={"height": "320px"}, config={"scrollZoom": True})],
                ),
            ],
        ),

        # Heatmap + feature importance
        html.Div(
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={"flex": "1", "minWidth": "620px"},
                    children=[dcc.Graph(id="heat_graph", style={"height": "320px"}, config={"scrollZoom": True})],
                ),
                html.Div(
                    style={"flex": "1", "minWidth": "520px"},
                    children=[dcc.Graph(id="imp_graph", style={"height": "320px"}, config={"scrollZoom": True})],
                ),
            ],
        ),

        html.Div(
            style={"marginTop": "8px", "color": "#444"},
            children="Brushing & linking: zoom the time series OR lasso points in Pred vs Actual → all charts update.",
        ),
    ],
)

# ----------------------------
# Reset: resets filters + lasso selection + forces zoom reset via reset_key/uirevision
# ----------------------------
@app.callback(
    Output("reset_key", "data"),
    Output("month_slider", "value"),
    Output("cbwd_dropdown", "value"),
    Output("scatter_graph", "selectedData"),
    Input("reset_btn", "n_clicks"),
    State("cbwd_dropdown", "options"),
    prevent_initial_call=True
)
def do_reset(n_clicks, cbwd_opts):
    all_cbwd = [o["value"] for o in cbwd_opts]
    return n_clicks, [1, 12], all_cbwd, None

# ----------------------------
# Main update (all figures + KPIs)
# ----------------------------
@app.callback(
    Output("resid_scatter", "figure"),
    Output("resid_hist", "figure"),
    Output("ts_graph", "figure"),
    Output("scatter_graph", "figure"),
    Output("cbwd_box", "figure"),
    Output("heat_graph", "figure"),
    Output("imp_graph", "figure"),
    Output("kpi_n", "children"),
    Output("kpi_r2", "children"),
    Output("kpi_rmse", "children"),
    Output("kpi_mae", "children"),
    Input("month_slider", "value"),
    Input("cbwd_dropdown", "value"),
    Input("ts_graph", "relayoutData"),
    Input("scatter_graph", "selectedData"),
    Input("reset_key", "data"),
)
def update_all(months_range, cbwd_list, relayoutData, selectedData, reset_key):
    # If reset was the trigger, ignore brush/selection this turn
    if ctx.triggered_id == "reset_key":
        relayoutData = None
        selectedData = None

    # 1) global filters
    d0 = apply_global_filters(df, months_range, cbwd_list)

    # 2) time brush (zoom)
    d1 = apply_time_brush(d0, relayoutData)

    # 3) lasso/box selection from Pred vs Actual
    d_final = apply_scatter_selection(d1, selectedData)

    # KPIs
    m = calc_metrics(d_final)
    kpi_n = f"Rows: {m['n']}"
    kpi_r2 = f"R²: {m['r2']:.3f}" if np.isfinite(m["r2"]) else "R²: —"
    kpi_rmse = f"RMSE: {m['rmse']:.2f}" if np.isfinite(m["rmse"]) else "RMSE: —"
    kpi_mae = f"MAE: {m['mae']:.2f}" if np.isfinite(m["mae"]) else "MAE: —"

    # ----------------------------
    # TOP: Residual scatter
    # ----------------------------
    # sample for speed/clarity
    n_sc = min(9000, len(d_final))
    d_res_sc = d_final.sample(n_sc, random_state=42) if len(d_final) > n_sc else d_final

    resid_scatter = px.scatter(
        d_res_sc, x="y_pred", y="resid",
        title="Residuals vs Predicted (top view)",
    )
    resid_scatter.update_traces(marker=dict(size=6, opacity=0.18))
    resid_scatter.add_hline(y=0)
    resid_scatter.update_layout(
        xaxis_title="Predicted PM2.5",
        yaxis_title="Residual (Actual - Predicted)",
        margin=dict(l=40, r=20, t=50, b=40),
        uirevision=reset_key,  # reset zoom when reset_key changes
    )

    # Residual histogram
    resid_hist = px.histogram(d_final, x="resid", nbins=60, title="Residual distribution")
    resid_hist.add_vline(x=0)
    resid_hist.update_layout(
        xaxis_title="Residual (Actual - Predicted)",
        yaxis_title="Count",
        margin=dict(l=40, r=20, t=50, b=40),
        uirevision=reset_key,
    )

    # ----------------------------
    # Time series (daily mean)
    # ----------------------------
    daily = d1.groupby(d1["datetime"].dt.date)[["y_true", "y_pred"]].mean().reset_index()
    daily["date"] = pd.to_datetime(daily["datetime"])

    ts_fig = go.Figure()
    ts_fig.add_trace(go.Scatter(x=daily["date"], y=daily["y_true"], mode="lines", name="Actual (daily mean)"))
    ts_fig.add_trace(go.Scatter(x=daily["date"], y=daily["y_pred"], mode="lines", name="Predicted (daily mean)"))
    ts_fig.update_layout(
        title="Daily mean PM2.5 (zoom to brush & link)",
        xaxis_title="Date",
        yaxis_title="PM2.5",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        uirevision=reset_key,
    )

    # ----------------------------
    # Predicted vs Actual (lasso selectable)
    # ----------------------------
    n_pa = min(9000, len(d1))
    d_sc = d1.sample(n_pa, random_state=42) if len(d1) > n_pa else d1

    scatter_fig = px.scatter(
        d_sc, x="y_true", y="y_pred",
        custom_data=["row_id"],
        title="Predicted vs Actual (lasso/box select to link)",
    )
    scatter_fig.update_traces(marker=dict(size=6, opacity=0.15))
    lims = [
        np.percentile(np.r_[d_sc["y_true"], d_sc["y_pred"]], 1),
        np.percentile(np.r_[d_sc["y_true"], d_sc["y_pred"]], 99),
    ]
    scatter_fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="y=x"))
    scatter_fig.update_layout(
        xaxis_title="Actual PM2.5",
        yaxis_title="Predicted PM2.5",
        margin=dict(l=40, r=20, t=50, b=40),
        uirevision=reset_key,
    )

    # ----------------------------
    # Boxplot: abs_err by cbwd
    # ----------------------------
    cbwd_box = px.box(
        d_final, x="cbwd", y="abs_err",
        points=False,
        title="Absolute error by wind direction (cbwd)",
        category_orders={"cbwd": cbwd_options},
    )
    cbwd_box.update_layout(
        xaxis_title="cbwd",
        yaxis_title="Absolute error",
        margin=dict(l=40, r=20, t=50, b=40),
        uirevision=reset_key,
    )

    # ----------------------------
    # Heatmap: MAE by hour x month
    # ----------------------------
    if len(d_final) > 0:
        pivot = d_final.pivot_table(index="hour", columns="month", values="abs_err", aggfunc="mean").sort_index()
        heat_fig = px.imshow(
            pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            aspect="auto",
            title="Mean absolute error by hour × month",
        )
        heat_fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Hour",
            margin=dict(l=40, r=20, t=50, b=40),
            uirevision=reset_key,
        )
    else:
        heat_fig = go.Figure()
        heat_fig.update_layout(title="Mean absolute error by hour × month", uirevision=reset_key)

    # ----------------------------
    # Feature importance (static)
    # ----------------------------
    imp_fig = px.bar(
        imp, x="importance", y="feature", orientation="h",
        title="Top feature importance (permutation)",
    )
    imp_fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="",
        margin=dict(l=40, r=20, t=50, b=40),
        uirevision=reset_key,
    )

    return (
        resid_scatter, resid_hist, ts_fig, scatter_fig, cbwd_box, heat_fig, imp_fig,
        kpi_n, kpi_r2, kpi_rmse, kpi_mae
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)

