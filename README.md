## Genetic Algorithm

The genetic algorithm is a heuristic search algorithm that mimics the biological reproduction and natural selection process belonging to the evolutionary algorithm class. In nature, individuals compete for resources such as food, water, and shelter. At the same time, each individual wants to continue its breed. Under these conditions, individuals who provide the best harmony to the environment win the race. In this way, they can continue their lineage. New individuals produced by the individuals who win the race also get their features from their ancestors. Evolutionary algorithms often produce good solutions to a variety of problems because they have no assumptions about which is the best. Genetic algorithms have been used to find solutions to problems such as traveling salesman problems (TSP). All these algorithms are built on a random process, and instead of producing a single solution to problems with limited information, they create a solution set consisting of different solutions. This cluster is named as the population in GA terminology.

### Concepts of Genetic Algorithm

**Gene:** City parameter (coordinates (x, y) of chromosome)<br/>
**Chromosome:** Creating a route by combining genes<br/>
**Fitness Function:** The function that allows a good route selection for the solved problem in the algorithm<br/>
**Population:** Chromosome set that can be selected to raise the next generation<br/>
**Crossover:** In the combination of chromosomes, offspring are obtained for future generations.<br/>
**Mutation:** exchange between two randomly selected cities (city is represented the gene in our algorithm)<br/>

### Objective

This project aims to solve the routing problem taken into consideration by minimizing the total distance traveled by garbage trucks of different capacities. The information of garbage level in the garbage container is obtained by sensors inside in garbage containers and the best route, consisting of the locations of the fully-loaded trash cans, is created.<br/>
Our problem is similar to the Travelling Salesman Problem (TSP) because we want to find the shortest way too, by visiting each trash can only once. Genetic algorithm is suggested to solve the TSP problem. <br/>

### TSP
The genetic algorithm finds a solution to the TSP problem. In this project, we use the solution of TSP to solve our problem. We have a list of 52 points (coordinate of waste bins) in the ```Berlin.txt```. The distance between each point is calculated by the program.<br/>
*In the genetic algorithm, there are usually 6 stages:*<br/>
-  Creating an initial population.<br/>
-  Calculating the fitness values of each chromosome in population.<br/>
-  Selecting the two best chromosomes by high fitness values.<br/>
-  Crossover to combine the genetic information of two parents (selected best chromosomes).<br/>
-  Mutating to increase variations.<br/>
-  Repeating until the best solution.<br/>
### IEEE

We published this project at IEEE.This link is below.<br/>
**IEEE link:** [IEEE](https://ieeexplore.ieee.org/document/9152865)

