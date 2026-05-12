import pandas as pd
import numpy as np

from geopy.distance import geodesic

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

# PARAMETERS
UAV_SPEED = 15
UAV_ALTITUDE = 100
BANDWIDTH = 1e6
NOISE = 1e-13
TX_POWER = 0.1

CPU_UAV = 1.2e9
CPU_UE = 4e8

# DATA PREPROCESSING
def extract_coordinates(point_text):

    point_text = point_text.replace("POINT(", "")
    point_text = point_text.replace(")", "")

    lon, lat = point_text.split()

    return float(lat), float(lon)


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
def calculate_latency(row):

    d = row["distance_km"] * 1000

    channel_gain = 1 / (
        d**2 + UAV_ALTITUDE**2
    )

    transmission_rate = BANDWIDTH * np.log2(
        1 + ((TX_POWER * channel_gain) / NOISE)
    )

    transmission_delay = (
        row["offload_ratio"]
        * row["task_size"]
    ) / transmission_rate

    edge_delay = (
        row["offload_ratio"]
        * row["task_size"]
    ) / CPU_UAV

    local_delay = (
        (1 - row["offload_ratio"])
        * row["task_size"]
    ) / CPU_UE

    total_delay = max(
        transmission_delay + edge_delay,
        local_delay
    )

    return total_delay

# ENERGY MODEL
def calculate_energy(row):

    d = row["distance_km"] * 1000

    flight_energy = 0.05 * d

    computation_energy = (
        row["offload_ratio"]
        * row["task_size"]
        * 1e-8
    )

    communication_energy = (
        TX_POWER
        * row["latency"]
    )

    total_energy = (
        flight_energy
        + computation_energy
        + communication_energy
    )

    return total_energy

# LOAD DATASET
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
    ]] = pd.DataFrame(start_coords.tolist())

    df[[
        "end_lat",
        "end_lon"
    ]] = pd.DataFrame(end_coords.tolist())

    df["distance_km"] = df.apply(
        calculate_distance,
        axis=1
    )

    df = df[df["distance_km"] > 0]

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

    df["latency"] = df.apply(
        calculate_latency,
        axis=1
    )

    df["energy"] = df.apply(
        calculate_energy,
        axis=1
    )

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    return df

# NSGA-II PROBLEM
class UAVMECProblem(Problem):

    def __init__(self, data):

        super().__init__(
            n_var=1,
            n_obj=3,
            n_constr=0,
            xl=0,
            xu=len(data) - 1,
            type_var=int
        )

        self.data = data

    def _evaluate(self, X, out, *args, **kwargs):

        f1 = []
        f2 = []
        f3 = []

        for row in X:

            idx = int(row[0])

            f1.append(
                self.data.iloc[idx]["latency"]
            )

            f2.append(
                self.data.iloc[idx]["energy"]
            )

            f3.append(
                -self.data.iloc[idx]["offload_ratio"]
            )

        out["F"] = np.column_stack([
            f1,
            f2,
            f3
        ])

# RUN NSGA-II
def run_nsga2(df):

    problem = UAVMECProblem(df)

    algorithm = NSGA2(
        pop_size=100
    )

    result = minimize(
        problem,
        algorithm,
        ('n_gen', 100),
        seed=42,
        verbose=True
    )

    return result