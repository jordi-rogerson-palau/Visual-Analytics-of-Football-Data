import duckdb
import pandas as pd
import altair as alt
import streamlit as st
from _constants import DB_PATH, LEAGUE_ORDER, LEAGUE_COLORS, HIDE_UI_CSS

st.markdown(HIDE_UI_CSS, unsafe_allow_html=True)
st.markdown("### Passes per Sequence vs Upfield Speed")

with st.expander("ℹ️ Information on the chart"):
    st.markdown(
        "Each point represents a league's average sequence profile across the 2015/16 season. "
        "The **x-axis** measures the average number of passes per possession sequence. "
        "The **y-axis** measures the average speed at which teams advance the ball upfield "
        "(negative values indicate sequences that go backwards). "
        "The **dashed lines** mark the overall mean for each axis, dividing the chart into four quadrants. "
        "Quadrant background colours follow a bivariate scheme to make the positioning more intuitive."
    )


@st.cache_data
def load_data():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df  = con.execute("SELECT * FROM sequences_league").df()
    con.close()
    return df


df = load_data()
df["league_name"] = pd.Categorical(df["league_name"], categories=LEAGUE_ORDER, ordered=True)

mean_passes = df["num_passes"].mean()
mean_speed  = df["average_speed"].mean()
INF         = 9999

X_DOM = [4.85, 5.2]
Y_DOM = [1.6,  1.85]


def _quad(x1, x2, y1, y2, hex_color):
    return (
        alt.Chart(pd.DataFrame({"x1": [x1], "x2": [x2], "y1": [y1], "y2": [y2]}))
        .mark_rect(color=hex_color, opacity=0.22)
        .encode(
            x=alt.X("x1:Q", scale=alt.Scale(domain=X_DOM)), x2="x2:Q",
            y=alt.Y("y1:Q", scale=alt.Scale(domain=Y_DOM)), y2="y2:Q",
        )
    )


quads = [
    _quad(-INF,        mean_passes, -INF,        mean_speed, "#e8e8e8"),
    _quad(mean_passes, INF,         -INF,        mean_speed, "#8fb3d0"),
    _quad(-INF,        mean_passes,  mean_speed, INF,        "#d0a06e"),
    _quad(mean_passes, INF,          mean_speed, INF,        "#7a6b5a"),
]

base   = alt.Chart(df)
points = base.mark_point(size=120, filled=True).encode(
    x=alt.X("num_passes:Q", scale=alt.Scale(domain=X_DOM), title="Avg Passes per Sequence"),
    y=alt.Y("average_speed:Q", scale=alt.Scale(domain=Y_DOM), title="Avg Upfield Speed On-Ball"),
    color=alt.Color("league_name:N",
                    scale=alt.Scale(domain=LEAGUE_ORDER,
                                    range=[LEAGUE_COLORS[l] for l in LEAGUE_ORDER]),
                    legend=alt.Legend(title="League")),
    tooltip=[
        "league_name:N",
        alt.Tooltip("num_passes:Q",    format=".2f"),
        alt.Tooltip("average_speed:Q", format=".2f"),
    ],
)

vline = base.transform_aggregate(mean_p="mean(num_passes)").mark_rule(
    strokeDash=[5, 5], color="#888888").encode(x="mean_p:Q")
hline = base.transform_aggregate(mean_s="mean(average_speed)").mark_rule(
    strokeDash=[5, 5], color="#888888").encode(y="mean_s:Q")

chart = (
    alt.layer(*quads, points, vline, hline)
    .properties(width=600, height=400)
    .configure_view(strokeWidth=0)
    .configure_axis(grid=False)
    .interactive()
)

col, _ = st.columns([1, 1])
with col:
    st.altair_chart(chart, use_container_width=False)
