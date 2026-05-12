import numpy as np

# FITNESS FUNCTION
def evaluate_solution(df, idx):

    row = df.iloc[idx]

    latency = row["latency"]

    energy = row["energy"]

    offload = -row["offload_ratio"]

    return np.array([
        latency,
        energy,
        offload
    ])

# DOMINATION CHECK
def dominates(a, b):

    return np.all(a <= b) and np.any(a < b)

# MOPSO
class Particle:

    def __init__(self, n_data):

        self.position = np.random.randint(
            0,
            n_data
        )

        self.velocity = np.random.uniform(
            -5,
            5
        )

        self.best_position = self.position

        self.best_fitness = None

# RUN MOPSO
def run_mopso(

    df,

    n_particles=100,

    n_iterations=100
):

    particles = [

        Particle(len(df))

        for _ in range(n_particles)
    ]

    global_best = None

    global_best_fitness = None

    pareto_archive = []

    for iteration in range(n_iterations):

        for particle in particles:

            idx = int(
                np.clip(
                    particle.position,
                    0,
                    len(df) - 1
                )
            )

            fitness = evaluate_solution(
                df,
                idx
            )

            # PERSONAL BEST

            if particle.best_fitness is None:

                particle.best_fitness = fitness

                particle.best_position = idx

            elif dominates(
                fitness,
                particle.best_fitness
            ):

                particle.best_fitness = fitness

                particle.best_position = idx

            # GLOBAL BEST

            if global_best_fitness is None:

                global_best_fitness = fitness

                global_best = idx

            elif dominates(
                fitness,
                global_best_fitness
            ):

                global_best_fitness = fitness

                global_best = idx

            # PARETO ARCHIVE

            pareto_archive.append(
                (idx, fitness)
            )----
        # UPDATE PARTICLE

        for particle in particles:

            w = 0.5

            c1 = 1.5

            c2 = 1.5

            r1 = np.random.rand()

            r2 = np.random.rand()

            particle.velocity = (

                w * particle.velocity

                +

                c1 * r1 * (
                    particle.best_position
                    -
                    particle.position
                )

                +

                c2 * r2 * (
                    global_best
                    -
                    particle.position
                )
            )

            particle.position += particle.velocity

            particle.position = np.clip(
                particle.position,
                0,
                len(df) - 1
            )

        print(
            f"Iteration {iteration+1}/{n_iterations}"
        )

    unique_archive = {}

    for idx, fit in pareto_archive:

        unique_archive[idx] = fit

    indices = list(unique_archive.keys())

    fitness_values = np.array(
        list(unique_archive.values())
    )

    return indices, fitness_values