import numpy as np

from pymoo.core.problem import Problem

from pymoo.algorithms.moo.moead import MOEAD

from pymoo.util.ref_dirs import get_reference_directions

from pymoo.optimize import minimize

# UAV-MEC PROBLEM
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

    def _evaluate(
        self,
        X,
        out,
        *args,
        **kwargs
    ):

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

# RUN MOEA/D
def run_moead(df):

    problem = UAVMECProblem(df)

    ref_dirs = get_reference_directions(
        "das-dennis",
        3,
        n_partitions=12
    )

    algorithm = MOEAD(

        ref_dirs=ref_dirs,

        n_neighbors=15,

        prob_neighbor_mating=0.7
    )

    result = minimize(
        problem,
        algorithm,
        ('n_gen', 100),
        seed=42,
        verbose=True
    )

    return result