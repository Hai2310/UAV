import pandas as pd
import numpy as np

from geopy.distance import geodesic


# EXTRACT GPS COORDINATES
def extract_coordinates(point_text):

    point_text = point_text.replace(
        "POINT(",
        ""
    )

    point_text = point_text.replace(
        ")",
        ""
    )

    lon, lat = point_text.split()

    return float(lat), float(lon)


# DISTANCE
def calculate_distance(row):

    start = (
        row["start_lat"],
        row["start_lon"]
    )

    end = (
        row["end_lat"],
        row["end_lon"]
    )

    return geodesic(start, end).km


# LATENCY MODEL
def calculate_latency(
    row,
    bandwidth,
    noise,
    tx_power,
    altitude,
    cpu_uav,
    cpu_ue
):

    d = row["distance_km"] * 1000

    channel_gain = 1 / (
        d**2 + altitude**2
    )

    transmission_rate = bandwidth * np.log2(
        1 +
        (
            tx_power * channel_gain
        ) / noise
    )

    transmission_delay = (
        row["offload_ratio"]
        *
        row["task_size"]
    ) / transmission_rate

    edge_delay = (
        row["offload_ratio"]
        *
        row["task_size"]
    ) / cpu_uav

    local_delay = (
        (1 - row["offload_ratio"])
        *
        row["task_size"]
    ) / cpu_ue

    total_delay = max(
        transmission_delay + edge_delay,
        local_delay
    )

    return total_delay


# ENERGY MODEL
def calculate_energy(
    row,
    tx_power
):

    d = row["distance_km"] * 1000

    flight_energy = 0.05 * d

    computation_energy = (
        row["offload_ratio"]
        *
        row["task_size"]
        *
        1e-8
    )

    communication_energy = (
        tx_power
        *
        row["latency"]
    )

    return (
        flight_energy
        +
        computation_energy
        +
        communication_energy
    )


# LOAD + PROCESS DATA
def load_dataset(file_path):

    df = pd.read_csv(file_path)

    start_coords = df["start_point"].apply(
        extract_coordinates
    )

    end_coords = df["end_point"].apply(
        extract_coordinates
    )

    df[[
        "start_lat",
        "start_lon"
    ]] = pd.DataFrame(
        start_coords.tolist()
    )

    df[[
        "end_lat",
        "end_lon"
    ]] = pd.DataFrame(
        end_coords.tolist()
    )

    df["distance_km"] = df.apply(
        calculate_distance,
        axis=1
    )

    df = df[
        df["distance_km"] > 0
    ]

    return df


# FEATURE GENERATION
def generate_features(df):

    np.random.seed(42)

    df["task_size"] = np.random.randint(
        1_000_000,
        5_000_000,
        len(df)
    )

    df["offload_ratio"] = np.random.uniform(
        0.2,
        1.0,
        len(df)
    )

    return df