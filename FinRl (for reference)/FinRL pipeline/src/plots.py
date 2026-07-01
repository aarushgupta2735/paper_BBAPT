import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_cumulative_returns(time_ind, drl_results, baseline_cumprod, ml_predictions):
    fig = go.Figure()
    traces = [
        (drl_results["A2C"]["cumprod"], "A2C"),
        (drl_results["PPO"]["cumprod"], "PPO"),
        (baseline_cumprod, "DJIA"),
        (ml_predictions["LR"][2], "LR"),
        (ml_predictions["RF"][2], "RF"),
        (ml_predictions["DT"][2], "DT"),
        (ml_predictions["SVM"][2], "SVM"),
    ]
    for trace_data, name in traces:
        fig.add_trace(go.Scatter(x=time_ind, y=trace_data, mode="lines", name=name))

    fig.update_layout(
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
            font=dict(family="sans-serif", size=15, color="black"),
            bgcolor="White",
            bordercolor="white",
            borderwidth=2,
        ),
        title={"y": 0.85, "x": 0.5, "xanchor": "center", "yanchor": "top"},
        paper_bgcolor="rgba(1,1,0,0)",
        plot_bgcolor="rgba(1,1,0,0)",
        xaxis_title="Date",
        yaxis=dict(titlefont=dict(size=30), title="Cumulative Return"),
        font=dict(size=40),
    )
    fig.update_layout(font_size=20)
    fig.update_traces(line=dict(width=2))
    apply_axis_style(fig)
    fig.show()


def plot_sharpe_vs_correlation(positive_ratio, positive_ratio_multi):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=positive_ratio_multi["algo"],
            y=positive_ratio_multi["Sharpe Ratio"],
            name="Sharpe Ratio",
            marker_size=15,
            line_width=5,
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(
            x=positive_ratio_multi["algo"],
            y=positive_ratio_multi["avg_correlation_coefficient"],
            name="Multi-Step Average Correlation Coefficient",
            width=0.38,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=positive_ratio["algo"],
            y=positive_ratio["avg_correlation_coefficient"],
            name="Single-Step Average Correlation Coefficient",
            width=0.38,
        ),
        secondary_y=False,
    )
    fig.update_layout(
        paper_bgcolor="rgba(1,1,0,0)",
        plot_bgcolor="rgba(1,1,0,0)",
        legend=dict(yanchor="top", y=1.5, xanchor="right", x=0.95),
        font_size=15,
    )
    fig.update_xaxes(
        title_text="Model",
        showline=True,
        linecolor="black",
        showgrid=True,
        gridwidth=1,
        gridcolor="LightSteelBlue",
        mirror=True,
    )
    fig.update_yaxes(
        title_text="Average Correlation Coefficient",
        secondary_y=False,
        range=[-0.1, 0.1],
        showline=True,
        linecolor="black",
        showgrid=True,
        gridwidth=1,
        gridcolor="LightSteelBlue",
        mirror=True,
    )
    fig.update_yaxes(title_text="Sharpe Ratio", secondary_y=True, range=[-0.5, 2.5])
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="LightSteelBlue")
    fig.show()


def plot_score_histograms(performance_score, multi_performance_score):
    for scores in [performance_score, multi_performance_score]:
        fig = make_subplots(rows=2, cols=3)
        for idx, algo in enumerate(["A2C", "PPO", "DT", "LR", "SVM", "RF"]):
            row, col = divmod(idx, 3)
            fig.append_trace(
                go.Histogram(
                    x=scores[scores["algo"] == algo]["score"].values,
                    nbinsx=25,
                    name=algo,
                    histnorm="probability",
                ),
                row + 1,
                col + 1,
            )

        fig.update_xaxes(title_text="Correlation coefficient", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_layout(
            paper_bgcolor="rgba(1,1,0,0)",
            plot_bgcolor="rgba(1,1,0,0)",
            font=dict(size=18),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1),
        )
        apply_axis_style(fig)
        fig.show()


def apply_axis_style(fig):
    fig.update_xaxes(
        showline=True,
        linecolor="black",
        showgrid=True,
        gridwidth=1,
        gridcolor="LightSteelBlue",
        mirror=True,
    )
    fig.update_yaxes(
        showline=True,
        linecolor="black",
        showgrid=True,
        gridwidth=1,
        gridcolor="LightSteelBlue",
        mirror=True,
    )
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="LightSteelBlue")
