"""
This code is for demonstrating the simulated annealing optimization 
for the Travelling Salesman Problem. 

The original implementation of the algorithm is from Page 494 of 
Computational Physics by Mark Newman and this is a modified version 
of his work for the purpose of personal study and review of the algorithm.

2022/May/7
Andrew Yooeun Chun
"""
from __future__ import annotations
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import *


class Map:

    city_locations: np.ndarray[float]
    number_of_cities: int
    map_seed: int
    tau: float
    optimization_seed: int
    total_distance: float

    def __init__(self, n: int, seed: Optional[int] = None) -> None:
        """
        Initialize the map with <n> cities. The coordinates of the cities are randomly
        generated unless <seed> is given.

        The salesman visit the cities in the order of self.city_locations[0] ->
        self.city_locations[1] -> self.city_locations[2] -> .. -> self.city_locations[-1]

        Note that self.city_locations[0] == self.city_locations[-1] since we assume that
        the salesman will return to the first city after the last city.
        """

        if seed:
            random.seed(seed)

        self.map_seed = seed

        self.city_locations = np.empty([n + 1, 2], float)
        self.number_of_cities = n

        for i in range(n):
            self.city_locations[i, 0] = random.random()
            self.city_locations[i, 1] = random.random()
            self.city_locations[n] = self.city_locations[0]

        self.total_distance = self.get_total_distance()

        return None

    def get_total_distance(self) -> float:
        """
        Return the total distance the salesman needs to travel
        """

        distance = 0.0
        for i in range(self.number_of_cities):
            distance += np.linalg.norm(
                self.city_locations[i + 1] - self.city_locations[i]
            )
        return distance

    def update_total_distance(self) -> None:
        """
        Update self.total_distance
        """

        self.total_distance = self.get_total_distance()

        return None

    def plot_configuration(self, save_fig=False) -> None:
        """
        Plot the initial configuration of the cities
        """

        plt.figure()
        plt.title(
            f"N={self.number_of_cities} cities located at random points (seed:{self.map_seed})"
        )
        plt.ylim((0, 1))
        plt.xlim((0, 1))
        plt.scatter(self.city_locations[:, 0], self.city_locations[:, 1])

        if save_fig:
            plt.savefig(f"{self.number_of_cities}_cities_seed:{self.map_seed}.png")
        plt.show()

    def plot_initial_path(self, save_fig=False) -> None:
        """
        Plot the initial path (order) of visiting each cities
        """

        plt.figure()
        plt.title(
            f"Initial configuration of the path for N={self.number_of_cities} cities (seed:{self.map_seed})"
        )
        plt.text(0.72, 0.95, f"Distance={self.total_distance:.4f}")
        plt.ylim((0, 1))
        plt.xlim((0, 1))
        plt.plot(self.city_locations[:, 0], self.city_locations[:, 1], "-o")
        if save_fig:
            plt.savefig(
                f"{self.number_of_cities}_cities_initial_path_Seed_{self.map_seed}.png"
            )
        plt.show()

    def plot_optimized_path(self, save_fig=False) -> None:
        """
        Plot the optimized path (order) of visiting the cities
        """

        plt.figure()
        plt.title(
            f"Optimized configuration of the path for N={self.number_of_cities} cities \
            \n (Initialization Seed: {self.map_seed} Optimization Seed:{self.optimization_seed} tau={self.tau:.2e})"
        )
        plt.text(0.75, 0.95, f"Distance={self.total_distance:.4f}")
        plt.ylim((0, 1))
        plt.xlim((0, 1))
        plt.plot(self.city_locations[:, 0], self.city_locations[:, 1], "-o")
        if save_fig:
            plt.savefig(
                f"{self.number_of_cities}_cities_initial_seed_{self.map_seed}_ \
                optimization_seed_{self.optimization_seed}_tau_{self.tau}.png"
            )
        plt.show()

    def simulated_annealing_optimization(
        self, tau: float, Tmin: float, Tmax: float, seed: Optional[int] = None
    ) -> np.ndarray[np.ndarray[float]]:
        """
        Optimize the path (order) of visiting each cities by simulated annealing method given the
        time constant <tau>, minimum temperature <Tmin>, and the maximum (initial) temperature <Tmax>.

        Note that different <seed> can produce different results due to the random nature of the method.

        Returns the most optimized path (shortest distance) in the form of np.ndarray[float] which may be 
        different from self.city_locations after finish executing. 
        """

        self.tau = tau
        self.optimization_seed = seed

        N = self.number_of_cities

        print(f"Simulating with {N} Cities.")
        print(f"Initial Total Distance is {self.total_distance}")

        random.seed(seed)

        t = 0
        T = Tmax

        best_distance = self.total_distance
        best_config = None

        # Main loop
        while T > Tmin:

            # Cooling
            t += 1
            T = Tmax * np.exp(-t / tau)

            # Choose two cities to swap and make sure they are distinct
            i, j = random.randrange(1, N), random.randrange(1, N)
            while i == j:
                i, j = random.randrange(1, N), random.randrange(1, N)

            # Swap them and calculate the change in distance
            previous_distance = self.total_distance

            self.city_locations[i, 0], self.city_locations[j, 0] = (
                self.city_locations[j, 0],
                self.city_locations[i, 0],
            )
            self.city_locations[i, 1], self.city_locations[j, 1] = (
                self.city_locations[j, 1],
                self.city_locations[i, 1],
            )
            self.update_total_distance()

            deltaD = self.total_distance - previous_distance

            if self.total_distance < best_distance:
                best_config = self.city_locations

            # If the move is rejected, swap them back again
            if random.random() > np.exp(-deltaD / T):
                self.city_locations[i, 0], self.city_locations[j, 0] = (
                    self.city_locations[j, 0],
                    self.city_locations[i, 0],
                )
                self.city_locations[i, 1], self.city_locations[j, 1] = (
                    self.city_locations[j, 1],
                    self.city_locations[i, 1],
                )

                self.total_distance = previous_distance

            if t % 1000 == 0:
                print(
                    f"Iteration: {t} Distance: {self.total_distance:.4f} T:{T:.4f} (T_min:{Tmin})"
                )

        self.update_total_distance()
        print(
            f"Iteration: {t} Distance: {self.total_distance:.4f} T:{T:.4f} (T_min:{Tmin})"
        )

        return best_config


if __name__ == "__main__":

    # seeds for reproducibility
    map_seed = 3141
    optimization_seed = 5926

    # suggested parameters for N=10
    N = 10
    Tmin = 1e-3
    Tmax = 10.0
    tau = 1e4

    m = Map(N, map_seed)

    m.plot_configuration()
    m.plot_initial_path()
    m.simulated_annealing_optimization(
        tau=tau, Tmin=Tmin, Tmax=Tmax, seed=optimization_seed
    )
    m.plot_optimized_path()
