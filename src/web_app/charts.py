"""Plotly chart builders (single y-axis, colorblind-safe Okabe-Ito palette, thin marks)."""
from __future__ import annotations

import plotly.graph_objects as go

# Okabe-Ito: distinguishable under common colour-vision deficiencies. Fixed order, never cycled.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#7F7F7F"]
INK = "#4a4a4a"
GRID = "rgba(128,128,128,0.18)"


def _layout(fig: go.Figure, title: str, height: int = 340, yfmt: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickformat=yfmt)
    return fig


def allocation_donut(allocation: dict[str, float], title: str) -> go.Figure:
    labels = [k.replace("_", " ").title() for k in allocation]
    fig = go.Figure(
        go.Pie(
            labels=labels, values=list(allocation.values()), hole=0.55, sort=False,
            marker=dict(colors=PALETTE[: len(labels)], line=dict(width=2, color="rgba(255,255,255,0.9)")),
            textinfo="label+percent", hovertemplate="%{label}: %{percent}<extra></extra>",
        )
    )
    return _layout(fig, title, legend=False, height=320)


def weight_bars(labels: list[str], weights: list[float], title: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=labels, y=[w * 100 for w in weights], marker=dict(color=PALETTE[0], cornerradius=4),
            text=[f"{w:.0%}" for w in weights], textposition="outside", hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_traces(width=0.5)
    _layout(fig, title, legend=False, yfmt=".0f")
    fig.update_yaxes(title="% of portfolio", rangemode="tozero")
    return fig


def price_chart(dates: list[str], closes: list[float], symbol: str) -> go.Figure:
    import pandas as pd

    s = pd.Series(closes, index=pd.to_datetime(dates))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name=f"{symbol} close", line=dict(color=PALETTE[0], width=2)))
    for window, color in ((20, PALETTE[1]), (50, PALETTE[2])):
        if len(s) >= window:
            fig.add_trace(go.Scatter(x=s.index, y=s.rolling(window).mean(), name=f"{window}-day avg",
                                     line=dict(color=color, width=1.5, dash="dot")))
    fig.update_yaxes(title="Price ($)")
    return _layout(fig, f"{symbol} price with moving averages", height=380, yfmt="$,.0f")


def day_change_bars(rows: list[dict]) -> go.Figure:
    rows = sorted(rows, key=lambda r: r["change_pct"])
    fig = go.Figure(
        go.Bar(
            x=[r["change_pct"] for r in rows], y=[r["name"] for r in rows], orientation="h",
            marker=dict(color=[PALETTE[2] if r["change_pct"] >= 0 else PALETTE[5] for r in rows], cornerradius=4),
            text=[f"{r['change_pct']:+.2f}%" for r in rows], textposition="outside",
            hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
        )
    )
    fig.update_traces(width=0.55)
    _layout(fig, "Day change (%)", legend=False, height=300)
    fig.update_xaxes(zeroline=True, zerolinecolor=INK, ticksuffix="%")
    fig.update_yaxes(showgrid=False)
    return fig


def goal_fan(plan: dict) -> go.Figure:
    mc = plan["monte_carlo"]
    yrs = mc["years"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yrs, y=mc["p90"], name="Optimistic (90th pct)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=yrs, y=mc["p10"], name="Likely range (10th-90th pct)", fill="tonexty",
                             fillcolor="rgba(0,114,178,0.15)", line=dict(width=0)))
    fig.add_trace(go.Scatter(x=yrs, y=mc["p50"], name="Median outcome", line=dict(color=PALETTE[0], width=2)))
    fig.add_trace(go.Scatter(x=list(range(len(plan["deterministic_path"]))), y=plan["deterministic_path"],
                             name="Steady-return path", line=dict(color=PALETTE[1], width=1.5, dash="dot")))
    fig.add_hline(y=plan["target"], line=dict(color=PALETTE[5], width=1.5, dash="dash"),
                  annotation_text=f"Target ${plan['target']:,.0f}", annotation_position="top left")
    fig.update_xaxes(title="Years")
    fig.update_yaxes(title="Balance ($)")
    return _layout(fig, "Projected balance", height=400, yfmt="$,.0f")
