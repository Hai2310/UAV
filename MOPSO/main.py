import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from sklearn.preprocessing import MinMaxScaler

from data_processing import *

from optimization import *

# PLOTLY
pio.renderers.default = "browser"
# PARAMETERS
UAV_ALTITUDE = 100

BANDWIDTH = 1e6

NOISE = 1e-13

TX_POWER = 0.1

CPU_UAV = 1.2e9

CPU_UE = 4e8
# LOAD DATA
file_path = "sf_dataset.csv"

df = load_dataset(file_path)

df = generate_features(df)
# LATENCY
df["latency"] = df.apply(

    lambda row: calculate_latency(

        row,

        BANDWIDTH,

        NOISE,

        TX_POWER,

        UAV_ALTITUDE,

        CPU_UAV,

        CPU_UE
    ),

    axis=1
)
# ENERGY
df["energy"] = df.apply(

    lambda row: calculate_energy(
        row,
        TX_POWER
    ),

    axis=1
)
# CLEANING
df = df.drop_duplicates()

df = df.reset_index(drop=True)

print(df.head())
# HISTOGRAMS
sns.set_style("whitegrid")

plt.figure(figsize=(10, 5))

sns.histplot(
    df["latency"],
    bins=50
)

plt.title(
    "Latency Distribution"
)

plt.show()

# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

sns.histplot(
    df["energy"],
    bins=50
)

plt.title(
    "Energy Distribution"
)

plt.show()
# HEATMAP
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

plt.title(
    "Feature Correlation"
)

plt.show()
# NORMALIZATION
scaler = MinMaxScaler()

scaled_features = scaler.fit_transform(

    df[[
        "latency",
        "energy",
        "offload_ratio"
    ]]
)
# RUN MOPSO
indices, pareto_front = run_mopso(df)

print("\nOptimization Completed")
# 2D PARETO
plt.figure(figsize=(10, 7))

plt.scatter(

    pareto_front[:, 0],

    pareto_front[:, 1],

    alpha=0.7
)

plt.xlabel("Latency")

plt.ylabel("Energy")

plt.title(
    "MOPSO Pareto Front"
)

plt.grid(True)

plt.show()
# 3D PARETO
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

    title="MOPSO 3D Pareto Front",

    scene=dict(

        xaxis_title='Latency',

        yaxis_title='Energy',

        zaxis_title='Offloading Ratio'
    ),

    width=900,

    height=700
)

fig.show(renderer="browser")
# BEST SOLUTION
best_idx = np.argmin(

    pareto_front[:, 0]
    +
    pareto_front[:, 1]
)

best_solution = pareto_front[
    best_idx
]

print("\n===== BEST SOLUTION =====")

print(
    f"Latency: {best_solution[0]:.6f}"
)

print(
    f"Energy: {best_solution[1]:.6f}"
)

print(
    f"Offloading Ratio: {-best_solution[2]:.6f}"
)
# OPTIMAL ROUTES
optimal_routes = df.iloc[
    indices
].copy()

optimal_routes = optimal_routes[[

    "trajectory",

    "distance_km",

    "latency",

    "energy",

    "offload_ratio"
]]

print("\nTop Optimal Routes:\n")

print(
    optimal_routes.head(10)
)
# MAP
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
# SAVE
optimal_routes.to_csv(

    "optimal_routes_mopso.csv",

    index=False
)

print(
    "\nResults saved successfully!"
)