import random
import numpy as np

from NSGA_II.visualization import UAVEnvironment


class NSGA2:
    def __init__(self, pop_size=20, generations=50):
        self.pop_size = pop_size
        self.generations = generations

        self.env = UAVEnvironment()

    # fitness function
    def evaluate(self, solution):
        user_pos = self.env.users[solution["user"]]

        uav_pos = np.array([
            solution["uav_x"],
            solution["uav_y"]
        ])

        distance = np.linalg.norm(
            user_pos - uav_pos
        )

        # objective 1: latency
        latency = (
            distance / 10
            + 5 * solution["offload"]
        )

        # objective 2: energy
        energy = (
            distance * 0.5
            + 20 * solution["offload"]
        )

        return latency, energy

    # crossover
    def crossover(self, p1, p2):
        child = {
            "uav_x": (p1["uav_x"] + p2["uav_x"]) / 2,
            "uav_y": (p1["uav_y"] + p2["uav_y"]) / 2,
            "user": p1["user"],
            "offload": (
                p1["offload"] + p2["offload"]
            ) / 2
        }

        return child

    # mutation
    def mutation(self, solution):
        solution["uav_x"] += random.uniform(-5, 5)
        solution["uav_y"] += random.uniform(-5, 5)

        solution["offload"] += random.uniform(
            -0.1, 0.1
        )

        solution["offload"] = max(
            0,
            min(1, solution["offload"])
        )

        return solution

    # Pareto selection
    def pareto_front(self, population):
        front = []

        for p in population:
            dominated = False

            for q in population:
                if (
                    q["fitness"][0] <= p["fitness"][0]
                    and
                    q["fitness"][1] <= p["fitness"][1]
                    and
                    q["fitness"] != p["fitness"]
                ):
                    dominated = True
                    break

            if not dominated:
                front.append(p)

        return front

    def run(self):
        population = []

        # initialize population
        for _ in range(self.pop_size):
            sol = self.env.random_solution()

            sol["fitness"] = self.evaluate(sol)

            population.append(sol)

        # evolution
        for gen in range(self.generations):

            offspring = []

            while len(offspring) < self.pop_size:

                p1, p2 = random.sample(
                    population, 2
                )

                child = self.crossover(p1, p2)

                child = self.mutation(child)

                child["fitness"] = self.evaluate(
                    child
                )

                offspring.append(child)

            population.extend(offspring)

            population = self.pareto_front(
                population
            )

            population = population[
                :self.pop_size
            ]

            print(f"Generation {gen}")

        return population