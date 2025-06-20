from collections import Counter

import numpy as np
import random

from tqdm import tqdm
# from run import get_data
import os
import matplotlib.pyplot as plt
from util import global_util
from util.util import parse_args, copy_data_folder_to_output
from util.action_space import Dispatch_rule
from RL_utils.env import Env
from util.config import config

args = parse_args()
# data_patients,data_source=get_data(args)
data_patients = [
    ['A', 0.716666666666667, 1.8355, 0, 0],
    ['A', 1.13833333333333, 3.16333333333333, 0, 1],
    ['A', 0.216666666666667, 1.03333333333333, 0, 2],
    ['A', 0.416666666666667, 1.83333333333333, 0, 0],
    ['A', 0.206666666666667, 2.19, 0,2],
    ['A', 0.856388888888889, 4.16033333333333, 0, 1],
]
data_source = {
    'before': 3,
    'ORtype': 1,
    'after': 1,
    'A': [2, 1, 3, 1],
}
half_ch = 3 * len(data_patients)
# 初始化种群
def check_contr(individual):
    count = Counter(individual[half_ch:])
    for key, value in count.items():
        if value != 3:
            return False
    return True
def initialize_population(size):
    population = []
    for _ in range(size):
        OR_ch = []
        start_range = env.source_idx[0]
        end_range = env.source_idx[-1]
        for data in data_patients:
            OR_ch.append(random.randint(start_range[0], start_range[-1]))
            OR_range = env.source_idx[ord(data[0]) - ord('A') + 1]
            OR_ch.append(random.randint(OR_range[0], OR_range[-1]))
            OR_ch.append(random.randint(end_range[0], end_range[-1]))
        stage_ch = [i for i in range(0, len(data_patients)) for _ in range(3)]
        random.shuffle(stage_ch)
        population.append(OR_ch + stage_ch)

    return population


# 适应度函数
def fitness(individual, env: Env):
    env.reset()
    for idx, patient_idx in enumerate(individual[half_ch:]):
        try:
            stage = env.patients[patient_idx].stage
            or_idx = stage + 3 * patient_idx
            oridx = individual[or_idx]
            resource = env.Before_sources + env.Operation_rooms + env.After_sources
            resource = resource[oridx]

            _, done = env.step(patient_idx, resource)
            if done:
                cmax = env.cmax
                avg_start = env.avg_start()
                rwd=env.U()
                break
        except:
            cmax = env.cmax
            avg_start = env.avg_start()
            rwd=0
            break
    return rwd, cmax, avg_start


# 选择操作
def selection(population, fitness_values, size=100):
    min_fitness = min(fitness_values)
    if min_fitness >= 0:
        non_negative_fitness = fitness_values
    else:
        non_negative_fitness = [f - min_fitness for f in fitness_values]

    total_fitness = sum(non_negative_fitness)

    if total_fitness > 0:
        selection_probs = [f / total_fitness for f in non_negative_fitness]
    else:
        # 如果总适应度为0，均匀分配概率
        selection_probs = [1 / len(population)] * len(population)

    indices = np.random.choice(len(population), size=size, p=selection_probs)
    return [population[i] for i in indices]


def repair(offspring, parent2):
    # 随机选择一个点位
    point = random.randint(half_ch, len(offspring) - 1)

    # 统计点位后面每个数字的个数
    count_after_point = Counter(offspring[point + 1:])

    # 删除后段
    offspring = offspring[:point + 1]

    for gene in parent2:
        if gene in count_after_point and count_after_point[gene] > 0:
            offspring.append(gene)
            count_after_point[gene] -= 1

    return offspring


def crossover(parent1, parent2, depth=0):
    if depth >= 10:
        return parent1, parent2

    length = len(parent1)
    alleles1 = [None for _ in range(half_ch)]
    alleles2 = [None for _ in range(half_ch)]

    # 随机选择父母的等位基因
    for i in range(half_ch):
        if random.random() < 0.5:
            alleles1[i] = parent1[i]
            alleles2[i] = parent2[i]
        else:
            alleles1[i] = parent2[i]
            alleles2[i] = parent1[i]

    # 创建子代
    offspring1 = alleles1 + list(parent1[half_ch:])
    offspring2 = alleles2 + list(parent2[half_ch:])

    # 修复冲突
    repair(offspring1, parent2)
    repair(offspring2, parent1)

    if not check_contr(offspring1) or not check_contr(offspring2):
        return crossover(parent1, parent2, depth + 1)  # 递归调用时增加深度

    return offspring1, offspring2

# 变异操作
def mutate(individual, max_gene):
    def swap_mutate(individual):
        idx1, idx2 = random.sample(range(half_ch, 2 * half_ch), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

    if random.random() < 0.5:  # 50%的概率进行交换变异
        swap_mutate(individual)
    else:  # 其他变异方式
        new_ch = initialize_population(1)[0]
        i = random.randint(0, half_ch)
        individual[i] = new_ch[i]


# 遗传算法主函数

def genetic_algorithm(env: Env, population_size=100, generations=300, mutation_rate=0.1):
    max_gene = config.patient_choice_num * 2
    population = initialize_population(population_size)

    best_value = []
    avg_value = []
    best_individual = None

    for generation in range(generations):
        fitness_values = []
        c_max_values = []
        avg_start_values = []

        for ind in population:
            f, c, a = fitness(ind, env)
            fitness_values.append(f)
            c_max_values.append(c)
            avg_start_values.append(a)
        avg = sum(fitness_values) / len(fitness_values)
        avg_value.append(sum(fitness_values) / len(fitness_values))
        best_index = max(range(len(fitness_values)), key=lambda i: fitness_values[i])
        best_individual = population[best_index]
        best_value.append(fitness_values[best_index])

        # 打印最佳个体的适应度、c_max和avg_start
        best_c_max = c_max_values[best_index]
        best_avg_start = avg_start_values[best_index]
        print(f"Generation {generation}: Max Fitness: {fitness_values[best_index]}, "
              f"Best c_max: {best_c_max}, Best Avg Start: {best_avg_start}", 'avg:', avg)
        print(best_individual)
        population = selection(population, fitness_values)

        # 精英保留
        new_population = [best_individual]
        for _ in range((population_size // 2) - 1):
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                mutate(child1, max_gene)
            if random.random() < mutation_rate:
                mutate(child2, max_gene)
            new_population.extend([child1, child2])

        population = new_population

    return best_individual, max(best_value), best_value, avg_value


# 绘制迭代过程
def plot_iterations(best_lengths, avg):
    plt.plot(best_lengths, label='best')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Value')
    plt.title('Fitness Value Over Generations')
    plt.plot(avg, label='average')
    plt.show()


def show_best(individual, env: Env):
    env.reset()
    for idx, patient_idx in enumerate(individual[half_ch:]):
        try:
            stage = env.patients[patient_idx].stage
            or_idx = stage + 3 * patient_idx
            oridx = individual[or_idx]
            resource = env.Before_sources + env.Operation_rooms + env.After_sources
            resource = resource[oridx]

            _, done = env.step(patient_idx, resource)
            if done:
                cmax = env.cmax
                avg_start = env.avg_start()
                break
        except:
            cmax = env.cmax
            avg_start = env.avg_start()
            break

    from util.util import Gantt
    print(env.U())
    Gantt(env.Before_sources + env.Operation_rooms + env.After_sources)


# 示例
if __name__ == "__main__":
    args = parse_args()
    global_util.setup_logger()

    env = Env(args, data_source, data_patients)
    env.init_start_matrix()
    # bi=[0, 2, 10, 1, 2, 9, 1, 3, 8, 1, 4, 9, 0, 5, 9, 1, 5, 10, 0, 7, 10, 0, 6, 8, 5, 1, 6, 3, 2, 5, 7, 7, 3, 4, 7, 1, 4, 2, 0, 1, 5, 4, 2, 6, 0, 6, 0, 3]
    import time
    # show_best(bi,env)
    start_time = time.time()
    best_individual, best_value, best_lengths, avg = genetic_algorithm(env)
    end_time = time.time()

    # 计算运行时长
    elapsed_time = end_time - start_time
    print(f"程序运行时长: {elapsed_time:.2f} 秒")
    print(f"Best Path: {best_individual}")
    print(f"Path Length: {best_value}")
    show_best(best_individual, env)
    plot_iterations(best_lengths, avg)
    #
    #
    args = parse_args()
    best1 = []
    best2 = []
    for i in tqdm(range(20)):

        env = Env(args, data_source, data_patients)
        env.init_start_matrix()
        best_individual, best_value, best_lengths, avg = genetic_algorithm(env)
        best1.append(best_value)
    print("best1:", best1)
