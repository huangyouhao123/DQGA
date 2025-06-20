import random


def crossover(parent1, parent2,idx=None):
    cross_list = [crossover1,
                  crossover2,
                  crossover3,
                  crossover4
                  ]
    if idx is None:
        idx = random.randint(0, len(cross_list) - 1)
    return cross_list[idx](parent1, parent2)


def crossover1(parent1, parent2):
    length = len(parent1)
    crossover_point = random.randint(1, length - 1)  # 确保不在两端
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    return offspring1, offspring2


def crossover2(parent1, parent2):
    length = len(parent1)
    point1 = random.randint(1, length - 2)  # 第一个交叉点
    point2 = random.randint(point1 + 1, length - 1)  # 第二个交叉点
    offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return offspring1, offspring2


def crossover3(parent1, parent2, probability=0.5):
    length = len(parent1)
    offspring1 = []
    offspring2 = []
    for i in range(length):
        if random.random() < probability:
            offspring1.append(parent1[i])
            offspring2.append(parent2[i])
        else:
            offspring1.append(parent2[i])
            offspring2.append(parent1[i])
    return offspring1, offspring2


def crossover4(parent1, parent2):
    length = len(parent1)
    start_point = random.randint(0, length - 1)
    offspring1 = [None] * length
    offspring2 = [None] * length
    index = start_point
    while True:
        offspring1[index] = parent1[index]
        offspring2[index] = parent2[index]
        index = (index + 1) % length
        if index == start_point:
            break
    return offspring1, offspring2
