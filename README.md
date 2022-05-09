# Travelling-Salesman-Problem

The travelling salesman problem is an NP-hard problem that asks to find the shortest possible tour that visits each city and returns to the starting point.
`simulated_annealing_salesman.py` uses a numerical optimization simulated annealing to solve this problem.

# Example

N=10 cities were randomly initalized and plotted

```python
    # seeds for reproducibility
    map_seed = 3141
    optimization_seed = 5926

    # suggested parameters for N=10
    N = 10
    Tmin = 1e-3
    Tmax = 10.0
    tau = 1e4

    m = Map(N, map_seed)
```

```python
    m.plot_configuration()
```
<p align="center">
  <img src=https://user-images.githubusercontent.com/85460898/167329714-b2a6e553-904d-4819-b633-0beb622fdb43.png />
</p>

The initial (random) path of visiting each city gives us a distance of 5.3790

<p align="center">
  <img src=https://user-images.githubusercontent.com/85460898/167330152-118f94ac-f546-4d67-b662-64f84ef96569.png />
</p>


The simulated annealing then optimizes our path
```python
    m.simulated_annealing_optimization(
        tau=tau, Tmin=Tmin, Tmax=Tmax, seed=optimization_seed
    )
```
output:
```
Simulating with 10 Cities.
Initial Total Distance is 5.378980669350638
Iteration: 1000 Distance: 4.6580 T:9.0484 (T_min:0.001)
Iteration: 2000 Distance: 5.6375 T:8.1873 (T_min:0.001)
Iteration: 3000 Distance: 5.8575 T:7.4082 (T_min:0.001)
Iteration: 4000 Distance: 5.4123 T:6.7032 (T_min:0.001)

.
.
.

Iteration: 92000 Distance: 2.6440 T:0.0010 (T_min:0.001)
Iteration: 92104 Distance: 2.6440 T:0.0010 (T_min:0.001)
```

<p align="center">
  <img src=https://user-images.githubusercontent.com/85460898/167331128-386c311f-26be-4f41-81e9-7c0343d2f7d0.png />
</p>

Although the simulated annealing doesn't always guarantees the most optimal results due to the random nature of any Monte Carlo method, we can observe that the algorithm had managed to decrease the distance by more than a factor of 2.

# References

Newman, M. (2012). Chapter 10 Random Processes and Monte Carlo Methods. In _Computational physics_, Mark Newman. 





