import numpy as np
import random
import operator
import time
import csv

population_size = 1000
mutate_rate = 0.1
two_opt_rate = 0.001
generations = 1000


# Every city has x and y coordinate information, also no for index
class City:
    def __init__(self, x, y, no):
        self.x = x
        self.y = y
        self.no = no

    def cost(self, city):
        # start_time = time.time()
        from_index = self.no
        to_index = city.no
        # end_time = time.time()
        # print("cost fonksiyonun çalışma süresi: " + str(end_time-start_time))
        return matrix[from_index][to_index]


# Following code block is for transfering a TSPLIB file to a List
# Attention! TSPLIB file must be in same folder with .py file
infile = open('berlin52', 'r')

# Read instance header
Name = infile.readline().strip().split()[1] # NAME
FileType = infile.readline().strip().split()[1] # TYPE
Comment = infile.readline().strip().split()[1] # COMMENT
Dimension = infile.readline().strip().split()[1] # DIMENSION
EdgeWeightType = infile.readline().strip().split()[1] # EDGE_WEIGHT_TYPE
infile.readline()

# Read node list
nodelist = []

# N means city length
# for berlin52 N should be 52, for 51city N should be 51
N = 52
for i in range(0, N):
    x, y = infile.readline().strip().split()[1:]
    nodelist.append([float(x), float(y)])

# Close input file
infile.close()

# Transform array to object array - city array
city_list = []

for i in range(len(nodelist)):
    city_list.append(City(x=nodelist[i][0], y=nodelist[i][1], no=i))


# city matrix is calculated once at the beginning
# city to city costs will be calculated via this matrix in order to decrease computation time
def city_matrix(city_list):
    start_time = time.time()
    matrix = []
    for i in range(0,len(city_list)):
        matrix.append([])
        for j in range(0,len(city_list)):
            xDis = abs(city_list[i].x - city_list[j].x)
            yDis = abs(city_list[i].y - city_list[j].y)
            cost = np.sqrt((xDis ** 2) + (yDis ** 2))
            matrix[i].append(cost)
    end_time = time.time()
    print("city_matrix fonksiyonun çalışma süresi: " + str(end_time-start_time))
    return matrix


# In order to use matrix in cost function, it should be created
matrix = city_matrix(city_list=city_list)


class Chromosome:  # Chromosome class for cost and fitness calculations
    def __init__(self, chromosome):  # chromosome parameter should be integer list
        # start_time = time.time()
        self.chromosome = chromosome
        self.no = 0
        self.cost = 0
        self.fitness = 0.0
        # end_time = time.time()
        # print("Chromosome init  :  " + str(end_time-start_time))

    def chromosome_cost(self):
        # start_time = time.time()
        if self.cost == 0:
            chromosome_cost = 0
            for i in range(0, len(self.chromosome)):
                from_city = self.chromosome[i]
                to_city = None
                if i + 1 < len(self.chromosome):
                    to_city = self.chromosome[i + 1]

                else:  # else means that this city is the last one, therefore next to_city should be the first city
                    to_city = self.chromosome[0]
                chromosome_cost += matrix[from_city][to_city]
            self.cost = chromosome_cost
        #end_time = time.time()
        #print("Chromosome cost  :  " + str(end_time - start_time))
        return self.cost

    def chromosome_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.chromosome_cost())
        return self.fitness


def create_chromosome(city_list):
    chromosome_object_list = random.sample(city_list, len(city_list))
    chromosome = []
    for i in range(0,len(chromosome_object_list)):
        chromosome.append(chromosome_object_list[i].no)
    chromosome = Chromosome(chromosome)
    chromosome.cost = chromosome.chromosome_cost()
    chromosome.fitness = 1 / float(chromosome.cost)
    return chromosome


def initial_population(pop_size, city_list):
    # start_time = time.time()
    population = []
    for i in range(0, pop_size):
        candidate_chromosome = create_chromosome(city_list)
        candidate_chromosome.no = i
        if candidate_chromosome not in population:
        # if created chromosome's cost exist in population, do not add it
            population.append(candidate_chromosome)
        else:
            i = i-1
    # end_time = time.time()
    # print("initial_population function çalışma süresi -- > " + str(end_time - start_time))
    return population


# sort population according to fitness value, from Highest to Lowest
def rank_chromosomes(population):
    fitness_results = {}
    for i in range(0, len(population)):
        fitness_results[i] = Chromosome(population[i]).chromosome_fitness()
    return sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=True)


def chromosomes2(population):
    cost_results = []
    for i in range(0,len(population)):
        cost_results[i] = population[i].cost
    return cost_results

# sort population according to cost value, from Highest to Lowest
def rank_cost(population):
    cost_results = {}
    for i in range(0, len(population)):
        cost_results[i] = population[i].chromosome_cost()
    return sorted(cost_results.items(), key=operator.itemgetter(1), reverse=False)


# list population without order according to their cost
def chromosomes(population):
    fitness_results = {}
    for i in range(0,len(population)):
        fitness_results[i] = Chromosome(population[i]).chromosome_cost()
    return random.sample(fitness_results.items(), len(fitness_results))


# Selection - Tournament
def selection_tournament(population):
    # start_time = time.time()
    # selection_results = []
    # print("pop_nonranked :  " + str(len(pop_nonranked)))
    ticket_1 = int(random.random() * len(population))  # Tournament capacity is four
    ticket_2 = int(random.random() * len(population))
    ticket_3 = int(random.random() * len(population))
    ticket_4 = int(random.random() * len(population))
    population[ticket_1].chromosome_cost()
    population[ticket_2].chromosome_cost()
    population[ticket_3].chromosome_cost()
    population[ticket_4].chromosome_cost()
    if population[ticket_1].cost < population[ticket_2].cost:
        winner = population[ticket_1]
    else:
        winner = population[ticket_2]
    if winner.cost < population[ticket_3].cost:
        winner = winner
    else:
        winner = population[ticket_3]
    if winner.cost < population[ticket_4].cost:
        winner = winner
    else:
        winner = population[ticket_4]
    parent = winner
    # end_time = time.time()
    # print("selection tournament function çalışma süresi -- > " + str(end_time - start_time))
    return parent.chromosome


# Breed
def breed(parent1, parent2):
    # start_time = time.time()
    child_p1 = []
    child_p2 = []

    gene_a = int(random.random() * len(parent1))
    gene_b = int(random.random() * len(parent1))

    start_gene = min(gene_a, gene_b)
    end_gene = max(gene_a, gene_b)

    # select randomly a subset of the first parent string
    # then fill remain chromosome with genes from second parent in order
    for i in range(start_gene, end_gene):
        child_p1.append(parent1[i])
    # in order not to duplicate, we use below command
    child_p2 = [item for item in parent2 if item not in child_p1]
    child = child_p1 + child_p2
    # end_time = time.time()
    # print("breed function çalışma süresi -- > " + str(end_time - start_time))
    return child


# Mutate
# in order not to drop any city, we use swap mutation
def mutate(chromosome, mutation_rate):
    # start_time = time.time()
    for swapped in range(len(chromosome)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(chromosome))

            city1 = chromosome[swapped]
            city2 = chromosome[swap_with]

            chromosome[swapped] = city2
            chromosome[swap_with] = city1
    # end_time = time.time()
    # print("mutate function çalışma süresi -- > " + str(end_time - start_time))
    return chromosome


# creating children till population size
def create_children(population, mutate_rate, two_opt_rate):
    #start_time = time.time()
    children = []
    children_costs = []
    elitism_index = rank_cost(population)
    elitism_index_1 = rank_cost(population)[0][0]
    elitism_index_2 = rank_cost(population)[1][0]
    children.append(population[elitism_index_1])
    children.append(population[elitism_index_2])

    # while len(children) == len(pop_nonranked):
    limit = len(population) - 2
    for j in range(0, limit):
        #first = time.time()
        #parent1_index = selection_tournament(pop_nonranked)
        #parent2_index = selection_tournament(pop_nonranked)
        #start_time = time.time()
        parent1 = selection_tournament(population)
        #end_time = time.time()
        #print("2 : " + str(end_time - start_time))
        #start_time = time.time()
        parent2 = selection_tournament(population)
        #end_time = time.time()
        #print("3 : " + str(end_time - start_time))
        #start_time = time.time()
        child = breed(parent1, parent2)
        #end_time = time.time()
        #print("4 : " + str(end_time - start_time))
        if random.random() < mutate_rate:
            #start_time = time.time()
            child_mutated = mutate(child, mutate_rate)
            #end_time = time.time()
            #print("5 : " + str(end_time - start_time))
        else:
            child_mutated = child

        isthere = 0
        chromosome_child = Chromosome(child_mutated)
        if random.random() < two_opt_rate:
            #print("bu child'a two opt uygulandı")
            chromosome_child = Chromosome(two_opt3(chromosome_child.chromosome))
        control_cost = chromosome_child.chromosome_cost()

        isthere = var(children_costs, control_cost)
        if isthere == 0:
            children.append(chromosome_child)
            children_costs.append(control_cost)
        else:
            #print("limit artırıldı")
            # print("--------------" + str(j) + ". child seçiminde aynı costa sahip kromozom bulunduğu için tekrar child seçiliyor")
            limit = limit + 1
        #end = time.time()
        #print("for döngüsü çalışma süresi : " + str(end-first))
    pop_check = len(population) - len(children)
    for j in range(0, pop_check):
        parent1 = selection_tournament(population)
        parent2 = selection_tournament(population)
        child = breed(parent1, parent2)
        if random.random() < mutate_rate:
            child_mutated = mutate(child, mutate_rate)
        else:
            child_mutated = child
        chromosome_child = Chromosome(child_mutated)
        if random.random() < 0.01:
            # print("bu child'a two opt uygulandı")
            chromosome_child = Chromosome(two_opt3(chromosome_child.chromosome))
        children.append(chromosome_child)
    # print("children uzunluğu: " + str(len(children)))
    #end_time = time.time()
    #print("create children çalışma süresi :  " + str(end_time-start_time))
    return children


def var(children_costs, control_cost):
    is_there = 0
    for i in range(0,len(children_costs)):
        if control_cost == children_costs[i]:
            is_there = 1
            return is_there
        else:
            is_there = 0
    return is_there



# Main code block
def genetic_algorithm(city_list, pop_size, two_opt_rate, mutate_rate, generations ):
    start_time = time.time()
    first_pop = initial_population(pop_size, city_list)
    print("Initial distance: --> " + str(first_pop[0].cost))

    # pop_nonranked = chromosomes(first_pop)
    evaluated_pop = first_pop
    #forcsv_end = []
    for i in range(0, generations):
        # evaluated_pop = create_generation(evaluated_pop, mutate_rate)
        #evaluate_time = time.time()
        evaluated_pop = create_children(evaluated_pop, mutate_rate, two_opt_rate)
        #print(str(i) + ". popülasyon uzunluğu - " + str(len(evaluated_pop)))
        #if random.random() < 0.01:
            #print("bu jenerasyonda two opt uygulandı")
            #evaluated_pop[0].chromosome = two_opt2(evaluated_pop[0].chromosome)
        #end_evaluate = time.time()
        #print("evaluate time : " + str(end_evaluate-evaluate_time))
        if i > 1:
            test = rank_cost(evaluated_pop)
            print(str(i) + ". iteration's best distance --> " + str(test[0][1]))
            print("-----average distance -->  " + str(test[49][1]))
            print("-----worst distance -->  "+ str(test[-1][1]))
        if i % 10 == 1:
            draw_path(evaluated_pop[0].chromosome)
        #forcsv = rank_cost(evaluated_pop)
        #forcsv_end.append(forcsv)
    best_route = rank_cost(evaluated_pop)[0]
    print("Final distance: --> " + str(best_route[1]))
    end_time = time.time()
    print("genetic_algorithm function çalışma süresi -- > " + str(end_time-start_time))
    best_index = rank_cost(evaluated_pop)[0][0]
    best = evaluated_pop[best_index]

    #mypath = "C:\Test"
    #csvfile = mypath + "/" + "test.csv"
    # Assuming res is a list of lists
    #with open(csvfile, "w") as output:
    #    writer = csv.writer(output, lineterminator='\n')
    #    writer.writerows(forcsv_end)

    return best


def two_opt(chromosome):
    best = chromosome
    for i in range(1, len(chromosome)-2):
        for j in range(i+1, len(chromosome)):
            if j-i == 1:continue # changes nothing, skip then
            new_chromosome = chromosome[:]
            new_chromosome[i:j] = chromosome[j-1:i-1:-1] # this is the 2optSwap
            new_cost = Chromosome(new_chromosome).chromosome_cost()
            best_cost = Chromosome(best).chromosome_cost()
            if(new_cost < best_cost): # what should cost be?
                best = new_chromosome

    return best


def two_opt3(chromosome):
    for i in range(0,len(chromosome)-2):
        #print("i değeri : " + str(i))
        for j in range(i+2, len(chromosome)-1):
            cost = matrix[chromosome[i]][chromosome[i+1]] + matrix[chromosome[j]][chromosome[j+1]]
            new_cost = matrix[chromosome[i]][chromosome[j]] + matrix[chromosome[i+1]][chromosome[j+1]]
            if new_cost < cost:
                new_chromosome = chromosome[:i+1]
                new_chromosome = new_chromosome + chromosome[j:i:-1]
                new_chromosome = new_chromosome + chromosome[j+1:]
                chromosome = new_chromosome
                #break
            else:
                new_chromosome = chromosome
    return new_chromosome


#pop = initial_population(100, city_list)
#test = pop[0]
#print(pop[0].chromosome_cost())
# print(test)
#test_chr = test.chromosome
#print(test_chr)
# print(test_chr[0])
# print(matrix[0][test_chr[0]])
# print(matrix[test_chr[0]][0])
#after_test = two_opt3(test_chr)
#print(after_test)
#print(Chromosome(after_test).chromosome_cost())



def two_opt2 (chromosome):
    best = chromosome
    improved = True
    while improved:
        improved = False
        for i in range(1, len(chromosome)-2):
            for j in range(i+1, len(chromosome)):
                if j-i == 1:continue # changes nothing, skip then
                new_chromosome = chromosome[:]
                new_chromosome[i:j] = chromosome[j-1:i-1:-1] # this is the 2optSwap
                #new_cost = Chromosome(new_chromosome).chromosome_cost()
                #best_cost = Chromosome(best).chromosome_cost()
                #if(new_cost < best_cost): # what should cost be?
                if Chromosome(new_chromosome).chromosome_cost() < Chromosome(best).chromosome_cost():  # what should cost be?
                    best = new_chromosome
                    improved = True
        chromosome = best
    return best


import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


def draw_path(route):
    # print(len(route))
    verts = []
    # verts = [(165,105), (95,187), (23,151), (14,128), (54,122), (57,67), (61,46), (152,12), (155,19), (197,76)]
    for i in range(0, len(route)):
        verts.append((city_list[route[i]].x, city_list[route[i]].y))
        #verts.append((route[i].x, route[i].y))
    # verts = []
    # for i in range(0,len(route)):
    #    verts.append(route[i])

    # print("grafik nokta uzunluğu : " + str(len(route)))

    # codes = []
    # codes.append(Path.MOVETO)
    # print(len(codes))
    # for j in range(0,len(route)):
    #   codes.append(Path.LINETO)

    # codes = codes.append(Path.CLOSEPOLY)

    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]

    # print(str(verts[0]) + " sol -- sag " + str(verts[50]))
    #start = verts[0]
    #end = verts[51]
    #x_values = [start[0], end[0]]
    #y_values = [start[1], end[1]]
    #plt.plot(x_values, y_values)
    # print(len(codes))
    path = Path(verts, codes)

    fig, ax = plt.subplots()
    patch = patches.PathPatch(path, facecolor='none', lw=2)
    ax.add_patch(patch)

    xs, ys = zip(*verts)
    ax.plot(xs, ys, 'o--', lw=2, color='black', ms=5)

    ax.set_xlim(-100, 1875)
    ax.set_ylim(-100, 1350)
    plt.show()


print("genetik algoritma başlıyor: ")

best = genetic_algorithm(city_list,population_size, two_opt_rate, mutate_rate, generations)
best = best.chromosome
best_of_best = two_opt3(best)
print("after n3 two opt3 is applied, final distance: ")
draw_path(best)
draw_path(best_of_best)
best_of_best = Chromosome(best_of_best)
print(best_of_best.chromosome_cost())
print("after n2 two opt is applied, final distance: ")
best_of_best = two_opt2(best)
best_of_best = Chromosome(best_of_best)
print(best_of_best.chromosome_cost())

def geneticAlgorithmPlot(population, pop_size, mutation_rate, two_opt_rate, generations):
    pop = initial_population(pop_size, population)
    progress = []
    progress.append(rank_cost(pop)[0][1])

    for i in range(0, generations):
        pop = create_children(pop, mutation_rate, two_opt_rate)
        progress.append(rank_cost(pop)[0][1])

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


geneticAlgorithmPlot(population=city_list, pop_size=100, mutation_rate=0.01, two_opt_rate=0.01, generations=100)


