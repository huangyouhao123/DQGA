import random
import tqdm
from tqdm import tqdm

import os
import matplotlib.pyplot as plt
from util import global_util
from util.util import parse_args, copy_data_folder_to_output
from util.action_space import Dispatch_rule
from RL_utils.env import Env
from util.config import config
from run import get_data
from collections import Counter
import numpy as np
import random
args = parse_args()
# data_patients,data_source=get_data(args)
# 26*(10+7+10)
data_patients = [
    ['A', 0.716666666666667, 1.8355, 0.665555555555556, 0],
    ['A', 1.13833333333333, 3.16333333333333, 0.749444444444444, 1],
    ['B', 0.666666666666667, 2.33333333333333, 0.583055555555556, 1],
    ['B', 0.216666666666667, 1.03333333333333, 1.08277777777778, 0],
    ['B', 0.666666666666667, 2.08333333333333, 0.414722222222222, 2],
    ['B', 0.416666666666667, 2.16666666666667, 0.575555555555556, 1],
    ['B', 0.416666666666667, 1.83333333333333, 0.662222222222222, 0],
    ['B', 0.433333333333333, 2.58333333333333, 0.415, 2],
    ['C', 0.206666666666667, 2.19, 0.568055555555556, 0],
    ['C', 0.856388888888889, 4.16033333333333, 0.910833333333333, 1],
    ['C', 0.123333333333333, 1.49, 0.582222222222222, 0],
    ['D', 0.963055555555556, 4.237, 0.989722222222222, 0],
    ['D', 0.0666666666666667, 2.66666666666667, 0.665277777777778, 1],
    ['E', 0.716666666666667, 1.8355, 0.665555555555556, 0],
    ['E', 1.13833333333333, 3.16333333333333, 0.749444444444444, 1],
    ['E', 0.416666666666667, 3.83333333333333, 0.825833333333333, 2],
    ['E', 1.03333333333333, 3.58333333333333, 0.413611111111111, 0],
    ['E', 1.21666666666667, 3.25083333333333, 0.818055555555556, 1],
    ['F', 0.633333333333333, 3.93283333333333, 0.735, 1],
    ['F', 0.85, 4.08333333333333, 0.745, 0],
    ['G', 0.5, 3.58333333333333, 1.07472222222222,0],
    ['G', 0.533333333333333, 1.43333333333333, 0.578055555555556,0],
    ['G', 0.316666666666667, 4.1825, 0.1525,1],
    ['G', 0.5, 4.26616666666667, 0.405,2],
    ['G', 0.866666666666667, 1.81666666666667, 0.826111111111111,2],
    ['G', 0.216666666666667, 4.23916666666667, 0.582777777777778,1],
]
data_source = {
    'before': 10,
    'ORtype': 7,
    'after': 10,
    'A': [2, 0.2, 2, 0.2],
    'B': [4, 0.2, 3, 0.2],
    'C': [2, 0.2, 2, 0.1],
    'D': [2, 0.1, 2, 0.2],
    'E': [3, 0.1, 3, 0.1],
    'F': [2, 0.1, 2, 0.1],
    'G': [4, 0.1, 3, 0.1],
}

env = Env(args, data_source, data_patients)
env.init_start_matrix()


class GeneticEnv:
    def __init__(self, num_of_all_patients):
        self.num_of_all_patients = num_of_all_patients
        self.action_space = list(range(0, 4))  # 假设有5种交叉方式
        self.last_value = 0

    def step(self, action, population, mutation_rate):
        population_size = len(population)

        new_population = []
        for _ in range(population_size):
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = crossover(parent1, parent2, action)
            if random.random() < mutation_rate:
                mutate(child1)
            if random.random() < mutation_rate:
                mutate(child2)
            new_population.extend([child1, child2])
        value = max([fitness(ind, env)[0] for ind in new_population])
        population = population + new_population
        # 计算适应度并给出奖励
        reward = value - self.last_value  # 奖励为第二高适应度与最高适应度的差值
        self.last_value = value
        return population, reward



class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=1, exploration_prob=1.0,
                 min_exploration_prob=0.1, decay_rate=0.9):
        self.q_table = np.zeros((4, 4))  # Q值表
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.min_exploration_prob = min_exploration_prob
        self.decay_rate = decay_rate
        self.action_space = action_space

    def choose_action(self, state):
        if np.random.rand() < self.exploration_prob:
            return np.random.choice(self.action_space)  # 探索
        return np.argmax(self.q_table[state])  # 利用

    def update_exploration_prob(self):
        self.exploration_prob = max(self.min_exploration_prob, self.exploration_prob * self.decay_rate)

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error


# 初始化种群
def initialize_population(size, length_of_chromosome, max_gene):
    population = []
    for _ in range(size):
        chromosome = [random.randint(0, max_gene - 1) for _ in range(length_of_chromosome)]
        population.append(chromosome)
    return population


# 适应度函数
def fitness(individual, env: Env):
    env.reset()

    for idx, action in enumerate(individual):
        patient_idx, resource = Dispatch_rule(action, env)
        _, done = env.step(patient_idx, resource)

        if done:
            cmax = env.cmax
            avg_start = env.avg_start()
            break

    return env.U(), cmax, avg_start


# 选择操作
def selection(population, fitness_values):
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

    indices = np.random.choice(len(population), size=100, p=selection_probs)
    return [population[i] for i in indices]


# 部分匹配交叉
from LSS import mutate
from CrossOver import crossover


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
    ep_reward = 0
    for idx, action in enumerate(individual):
        patient_idx, resource = Dispatch_rule(action, env)
        _, reward, done = env.step(patient_idx, resource)
        ep_reward += reward
        if done:
            cmax = env.cmax
            avg_start = env.avg_start()
            break
    from util.util import Gantt
    print(ep_reward)
    Gantt(env.Before_sources + env.Operation_rooms + env.After_sources)


def genetic_algorithm_with_rl(Gene_env: GeneticEnv, population_size=100, generations=300, mutation_rate=0.05,
                              elite_fraction=0.1,agent=None):
    max_gene = config.patient_choice_num * 2
    population = initialize_population(population_size, Gene_env.num_of_all_patients * 3, max_gene)
    if agent is None:
        agent = QLearningAgent(Gene_env.action_space)

    best_value = []
    avg_value = []
    best_individual = None
    action = 0
    state = 0
    action = agent.choose_action(state)
    state = action

    for generation in range(generations):
        action = agent.choose_action(state)
        next_state = action

        new_population, reward = Gene_env.step(action, population, mutation_rate)
        if generation != 0:
            agent.update(state, action, reward, next_state)

        # 更新种群
        fitness_values = [fitness(ind, env)[0] for ind in new_population]
        avg = sum(fitness_values) / len(fitness_values)
        avg_value.append(sum(fitness_values) / len(fitness_values))
        best_index = max(range(len(fitness_values)), key=lambda i: fitness_values[i])
        best_individual = new_population[best_index]
        best_value.append(fitness_values[best_index])

        # 精英保留策略
        elite_count = int(population_size * elite_fraction)
        elites = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i], reverse=True)[:elite_count]
        elite_individuals = [new_population[i] for i in elites]

        # 选择剩余个体
        population = selection(new_population, fitness_values)
        population += elite_individuals  # 保留精英个体

        state = next_state  # 更新状态
        print(f"Generation {generation}: Max Fitness: {fitness_values[best_index]} ", 'avg:', avg)
        print(best_individual)

    return best_individual, max(best_value), best_value, avg_value

if __name__ == '__main__':
    args = parse_args()
    best1 = []
    best2 = []
    agent = QLearningAgent(GeneticEnv(len(data_patients)).action_space)
    with open('output_qa.txt', 'w') as f:
        for i in tqdm(range(10)):
            gen_env = GeneticEnv(len(data_patients))
            import time
            # 记录开始时间

            # best_individual, best_value, best_lengths, avg = genetic_algorithm_with_rl(gen_env,agent=agent)
            best_individual, best_value, best_lengths, avg = genetic_algorithm_with_rl(gen_env)


            best2.append(best_value)
        f.write('best2: {}\n'.format(best2))

