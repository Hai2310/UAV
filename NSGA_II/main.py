import plotly.io as pio

from uav_mec_model import (
    load_dataset,
    run_nsga2
)

from visualization import (
    plot_distributions,
    plot_heatmap,
    plot_pareto,
    plot_3d_pareto,
    plot_map
)
# FIX PLOTLY
pio.renderers.default = "browser"
# LOAD DATA
df = load_dataset("sf_dataset.csv")

print(df.head())
# VISUALIZATION
plot_distributions(df)

plot_heatmap(df)
# RUN NSGA-II
result = run_nsga2(df)

pareto_front = result.F

print("\nOptimization Completed")

print("\nPareto Front:\n")

print(pareto_front[:10])
# PLOTS
plot_pareto(pareto_front)

plot_3d_pareto(pareto_front)

plot_map(df)
# BEST SOLUTION
best_solution_idx = pareto_front[:, 0].argmin()

best_solution = pareto_front[
    best_solution_idx
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
# SAVE RESULTS
optimal_indices = result.X.flatten().astype(int)

optimal_routes = df.iloc[
    optimal_indices
].copy()

optimal_routes.to_csv(
    "optimal_routes.csv",
    index=False
)

print("\nResults saved successfully!")