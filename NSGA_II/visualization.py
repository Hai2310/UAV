import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# DISTRIBUTIONS
def plot_distributions(df):

    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 5))

    sns.histplot(
        df["latency"],
        bins=50
    )

    plt.title("Latency Distribution")
    plt.xlabel("Latency")
    plt.ylabel("Frequency")

    plt.show()

    plt.figure(figsize=(10, 5))

    sns.histplot(
        df["energy"],
        bins=50
    )

    plt.title("Energy Distribution")
    plt.xlabel("Energy")
    plt.ylabel("Frequency")

    plt.show()

# HEATMAP
def plot_heatmap(df):

    plt.figure(figsize=(8, 6))

    sns.heatmap(

        df[[
            "distance_km",
            "latency",
            "energy",
            "offload_ratio"
        ]].corr(),

        annot=True,
        cmap="Blues"
    )

    plt.title("Feature Correlation")

    plt.show()

# PARETO FRONT
def plot_pareto(pareto_front):

    plt.figure(figsize=(10, 7))

    plt.scatter(
        pareto_front[:, 0],
        pareto_front[:, 1],
        alpha=0.7
    )

    plt.xlabel("Latency")
    plt.ylabel("Energy")

    plt.title("Pareto Front")

    plt.grid(True)

    plt.show()

# 3D PARETO
def plot_3d_pareto(pareto_front):

    fig = go.Figure(

        data=[

            go.Scatter3d(

                x=pareto_front[:, 0],
                y=pareto_front[:, 1],
                z=-pareto_front[:, 2],

                mode='markers',

                marker=dict(
                    size=5,
                    opacity=0.8
                )
            )
        ]
    )

    fig.update_layout(

        title="3D Pareto Front",

        scene=dict(

            xaxis_title='Latency',
            yaxis_title='Energy',
            zaxis_title='Offloading Ratio'
        ),

        width=900,
        height=700
    )

    fig.show(renderer="browser")

# MAP
def plot_map(df):

    sample_df = df.sample(1000)

    fig = px.scatter_mapbox(

        sample_df,

        lat="start_lat",
        lon="start_lon",

        hover_name="trajectory",

        hover_data=[
            "latency",
            "energy"
        ],

        zoom=10,
        height=700
    )

    fig.update_layout(
        mapbox_style="open-street-map"
    )

    fig.update_layout(
        title="UAV-MEC User Locations"
    )

    fig.show(renderer="browser")