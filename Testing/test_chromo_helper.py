from ModelHelper import *


import random





# random.seed(100)
chromosome = generate_chromosome()

for i in range(3000):
    chromosome = mutate_chromosome(chromosome)

print(i,":", chromosome, len(chromosome['layers']))
