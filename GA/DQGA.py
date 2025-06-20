import math
import os
import random
from collections import Counter
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from RL_utils.env import Env
from util.action_space import Dispatch_rule
from util.config import config
from util.util import parse_args

args = parse_args()
# data_patients, data_source = get_data(args)
data_patients = [
    ['A', 0.716666666666667, 1.8355, 0.665555555555556, 0],
    ['A', 1.13833333333333, 3.16333333333333, 0.749444444444444, 1],
    ['B', 0.216666666666667, 1.03333333333333, 1.08277777777778, 0],
    ['B', 0.416666666666667, 1.83333333333333, 0.662222222222222, 1],
    ['B', 0.433333333333333, 2.58333333333333, 0.415, 2],
    ['C', 0.206666666666667, 2.19, 0.568055555555556, 0],
    ['C', 0.856388888888889, 4.16033333333333, 0.910833333333333, 1],
    ['C', 0.123333333333333, 1.49, 0.582222222222222, 0],
]
data_source = {
    'before': 3,
    'ORtype': 3,
    'after': 3,
    'A': [2, 0.3, 2, 0.333],
    'B': [2, 0.4, 3, 0.333],
    'C': [2, 0.3, 2, 0.333]
}

env = Env(args, data_source, data_patients)
env.init_start_matrix()

half_ch = 3 * len(data_patients)

train_flag = True


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


def fitness(individual, env: Env):
    env.reset()
    for idx, patient_idx in enumerate(individual[half_ch:]):

        stage = env.patients[patient_idx].stage
        or_idx = stage + 3 * patient_idx
        oridx = individual[or_idx]
        resource = env.Before_sources + env.Operation_rooms + env.After_sources
        resource = resource[oridx]

        _, done = env.step(patient_idx, resource)
        if done:
            cmax = env.cmax
            avg_start = env.avg_start()
            rwd = env.U()
            break

    return rwd, cmax, avg_start


# 选择操作
def selection(population, fitness_values, size=100):
    min_fitness = min(fitness_values)
    if min_fitness<0:
        fitness_values=[i-min_fitness for i in fitness_values]
    non_negative_fitness=fitness_values
    total_fitness = sum(non_negative_fitness)

    if total_fitness > 0:
        selection_probs = [f / total_fitness for f in non_negative_fitness]
    else:
        # 如果总适应度为0，均匀分配概率
        selection_probs = [1 / len(population)] * len(population)

    indices = np.random.choice(len(population), size=size, p=selection_probs)
    return [population[i] for i in indices]


# 部分匹配交叉

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


def crossover_or(parent1, parent2, depth=0):
    def add_order(individual):
        order_dict=Counter(individual)
        weight=int(math.log10(len(individual))) + 1
        weight=pow(10,weight)
        for i in range(len(individual)):
            t=order_dict[individual[i]]
            order_dict[individual[i]] -= 1
            individual[i]=weight*individual[i]+t

        return individual

    def remove_order(individual):
        weight=int(math.log10(len(individual))) + 1
        weight=pow(10,weight)
        for i in range(len(individual)):
            individual[i]=individual[i]//weight
        return individual

    def crossover(parent1, parent2):
        length = len(parent1)
        # 随机确定两个交叉点C1和C2，且C1 < C2
        C1, C2 = sorted(random.sample(range(1, length + 1), 2))

        # 从parent1中选取C1和C2之间的基因段
        segment1 = parent1[C1 - 1:C2]
        segment2 = parent2[C1 - 1:C2]

        # 从parent2中提取不在C1和C2之间的基因
        offspring1 = [gene for gene in parent2 if gene not in segment1]
        offspring2 = [gene for gene in parent1 if gene not in segment2]

        # 将选取的基因段插入到提取后的基因列表中，形成offspring1
        offspring1[C1 - 1:C1 - 1] = segment1
        offspring2[C1 - 1:C1 - 1] = segment2

        return offspring1, offspring2

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

    # # 创建子代
    # offspring1 = alleles1 + list(parent1[half_ch:])
    # offspring2 = alleles2 + list(parent2[half_ch:])
    # 创建子代
    after1=parent1[half_ch:]
    after2=parent2[half_ch:]
    after1=add_order(after1)
    after2=add_order(after2)
    after1,after2=crossover(after1,after2)
    after1=remove_order(after1)
    after2=remove_order(after2)
    # 创建子代
    offspring1 = alleles1 + after1
    offspring2 = alleles2 + after2

    # # 修复冲突
    # repair(offspring1, parent2)
    # repair(offspring2, parent1)

    return offspring1, offspring2


# 变异操作
def mutate_or(individual, max_gene):
    def swap_mutate(individual):
        idx1, idx2 = random.sample(range(half_ch, 2 * half_ch), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

    # if random.random() < 0.5:  # 50%的概率进行交换变异
    #     swap_mutate(individual)
    # else:  # 其他变异方式
    #     new_ch = initialize_population(1)[0]
    #     i = random.randint(0, half_ch)
    #     individual[i] = new_ch[i]
    swap_mutate(individual)
    new_ch = initialize_population(1)[0]
    i = random.randint(0, half_ch-1)
    individual[i] = new_ch[i]



# 部分匹配交叉
from LSS import mutate
from CrossOver import crossover


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
        value = max([fitness_RL(ind, env)[0] for ind in new_population])
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
        self.exploration_prob = exploration_prob if train_flag else 0
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

    def save_q_table(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, file_path):
        with open(file_path, 'rb') as f:
            self.q_table = pickle.load(f)


# 初始化种群
def initialize_population_RL(size, length_of_chromosome, max_gene):
    population = []
    for _ in range(size):
        chromosome = [random.randint(0, max_gene - 1) for _ in range(length_of_chromosome)]
        population.append(chromosome)
    return population


# 适应度函数
def fitness_RL(individual, env: Env):
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
def selection_RL(population, fitness_values, size=100):
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


# 绘制迭代过程
def plot_iterations(best_lengths, avg):
    plt.plot(best_lengths, label='best')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Value')
    plt.title('hunhe')
    plt.plot(avg, label='average')
    plt.show()


def show_best_RL(individual, env: Env):
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
    Gantt(env.Before_sources + env.Operation_rooms + env.After_sources)


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
    Gantt(env.Before_sources + env.Operation_rooms + env.After_sources,env)


def rl_part(Gene_env, agent, generations1, population, best_value: list, best_individual, avg_value: list,
            elite_fraction=None, save_count=None, is_guide=False):
    population_size = len(population)

    elite_count = int(population_size * elite_fraction)
    if is_guide:
        elite_count = int(save_count)
    action = 0
    state = 0
    action = agent.choose_action(state)
    state = action

    for generation in range(generations1):
        action = agent.choose_action(state)
        next_state = action

        new_population, reward = Gene_env.step(action, population, 0.1)
        if generation != 0 and not train_flag:
            agent.update(state, action, reward, next_state)

        # 更新种群
        fitness_values = [fitness_RL(ind, env)[0] for ind in new_population]
        avg = sum(fitness_values) / len(fitness_values)
        avg_value.append(sum(fitness_values) / len(fitness_values))
        best_index = max(range(len(fitness_values)), key=lambda i: fitness_values[i])
        best_individual = new_population[best_index]
        best_value.append(fitness_values[best_index])

        # 精英保留策略

        elites = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i], reverse=True)[:elite_count]
        elite_individuals = [new_population[i] for i in elites]
        if generations1 == 1:
            return elite_individuals
        # 选择剩余个体
        population = selection_RL(new_population, fitness_values)
        population += elite_individuals  # 保留精英个体

        state = next_state  # 更新状态
        print(f"Generation {generation}: Max Fitness: {fitness_values[best_index]} ", 'avg:', avg)
        print(best_individual)
    return population.copy()


def genetic_algorithm_with_rl(Gene_env: GeneticEnv, population_size=100, generations1=2, generations2=100,
                              mutation_rate=0.1,
                              elite_fraction=0.1, rl_Saved=0.05, agent=None):
    max_gene = config.patient_choice_num * 2
    population = initialize_population_RL(population_size, Gene_env.num_of_all_patients * 3, max_gene)
    if agent is None:
        agent = QLearningAgent(Gene_env.action_space)
    best_value = []
    avg_value = []
    best_individual = None
    count = 0

    rl_population = rl_part(Gene_env, agent, generations1, population, best_value, best_individual, avg_value,
                            elite_fraction=elite_fraction)

    # population = [change_code(individual, env) for individual in rl_population]
    population = initialize_population(int(population_size * (1 - rl_Saved)))
    for generation in range(generations2):
        fitness_values = []
        c_max_values = []
        avg_start_values = []
        rl_population = rl_part(Gene_env, agent, 1, rl_population, [], best_individual, [],
                                elite_fraction=elite_fraction, save_count=rl_Saved * population_size, is_guide=True)
        rl_population_change = [change_code(individual, env) for individual in rl_population]
        hybrid_ch = []
        if count % config.hybrid_jump == 0:
            for normal, guide in zip(random.sample(population, len(rl_population_change)), rl_population_change):
                hybrid_ch.append(guide[:half_ch] + normal[half_ch:])
                hybrid_ch.append(normal[:half_ch] + guide[half_ch:])
        count += 1

        for _ in range((population_size // 2) - 1):
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = crossover_or(parent1, parent2)
            if random.random() < mutation_rate:
                mutate_or(child1, max_gene)
            if random.random() < mutation_rate:
                mutate_or(child2, max_gene)
            population.extend([child1, child2])
        population.extend(hybrid_ch)

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

        # 精英保留
        elite_count = int(population_size * elite_fraction)
        elites = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i], reverse=True)[:elite_count]
        elite_individuals = [population[i] for i in elites]
        population = selection(population, fitness_values)
        population = population + [best_individual] + elite_individuals

    return best_individual, max(best_value), best_value, avg_value


def change_code(individual, env: Env):
    env.reset()
    count = half_ch
    new_ch = [None for i in range(half_ch * 2)]
    for idx, action in enumerate(individual):
        patient_idx, resource = Dispatch_rule(action, env)
        _, done = env.step(patient_idx, resource)
        new_ch[count] = patient_idx
        count += 1
        if done:
            cmax = env.cmax
            avg_start = env.avg_start()
            break
    Machines = env.Before_sources

    for i in range(len(Machines)):
        for j in range(len(Machines[i].start)):
            idx = Machines[i]._on[j] * 3
            new_ch[idx] = i

    Machines = env.Operation_rooms
    for i in range(len(Machines)):
        for j in range(len(Machines[i].start)):
            new_ch[Machines[i]._on[j] * 3 + 1] = i + len(env.Before_sources)

    Machines = env.After_sources
    for i in range(len(Machines)):
        for j in range(len(Machines[i].start)):
            new_ch[Machines[i]._on[j] * 3 + 2] = i + len(env.Before_sources) + len(env.Operation_rooms)

    return new_ch


if __name__ == '__main__':
    # import time
    # gen_env = GeneticEnv(len(data_patients))
    # start_time = time.time()
    # best_individual, best_value, best_lengths, avg = genetic_algorithm_with_rl(gen_env)
    # # 记录结束时间
    # end_time = time.time()
    # # 计算运行时长
    # elapsed_time = end_time - start_time
    # print(f"程序运行时长: {elapsed_time:.2f} 秒")
    # # bi=[3, 6, 17, 3, 5, 20, 1, 7, 21, 0, 7, 17, 4, 8, 17, 1, 8, 20, 4, 7, 19, 2, 8, 17, 2, 10, 21, 2, 9, 18, 2, 9, 20, 0, 12, 18, 2, 11, 18, 3, 14, 21, 1, 13, 20, 1, 14, 17, 0, 14, 19, 2, 13, 18, 1, 16, 20, 4, 15, 19, 10, 3, 3, 12, 12, 12, 17, 17, 10, 17, 8, 3, 8, 10, 5, 5, 8, 1, 19, 6, 1, 11, 5, 6, 6, 9, 11, 15, 11, 18, 7, 0, 19, 7, 18, 0, 2, 0, 2, 9, 2, 9, 16, 19, 14, 1, 4, 13, 18, 15, 15, 13, 7, 13, 14, 14, 4, 16, 4, 16]
    # # show_best(bi,env)
    # print(f"Best Path: {best_individual}")
    # print(f"Path Length: {best_value}")
    # show_best(best_individual, env)
    # plot_iterations(best_lengths, avg)

    # all_generations = 300
    # tao_list = [0.02, 0.05, 0.07, 0.1]
    # hybrid_jump_list = [1, 3, 5, 10]
    # ge1rate_list = [0.050, 0.066, 0.100, 0.133]
    #
    # with open('output.txt', 'a') as f:  # 以追加模式打开文件
    #     for tao in tao_list:
    #         for hybrid in hybrid_jump_list:
    #             for ge1rate in ge1rate_list:
    #                 generation1 = int(all_generations * ge1rate + 0.5)
    #                 generation2 = all_generations - generation1
    #                 config.hybrid_jump = hybrid
    #
    #                 args = parse_args()
    #                 best1 = []
    #                 best2 = []
    #                 # agent = QLearningAgent(GeneticEnv(len(data_patients)).action_space)
    #
    #                 for i in tqdm(range(20)):
    #                     gen_env = GeneticEnv(len(data_patients))
    #                     best_individual, best_value, best_lengths, avg = genetic_algorithm_with_rl(
    #                         gen_env,generations1=generation1, generations2=generation2, rl_Saved=tao)
    #                     best2.append(best_value)
    #
    #
    #                 # 将结果写入文件
    #                 f.write(f'Tao: {tao}, Hybrid Jump: {hybrid}, Ge1rate: {ge1rate}, Best2: {best2}\n')
    #                 print(f'Tao: {tao}, Hybrid Jump: {hybrid}, Ge1rate: {ge1rate}, Best2: {best2}\n')


    best1 = []
    best2 = []
    agent = QLearningAgent(GeneticEnv(len(data_patients)).action_space)

    for i in tqdm(range(1)):
        if not train_flag:
            agent.load_q_table('q_table.pkl')
        gen_env = GeneticEnv(len(data_patients))
        best_individual, best_value, best_lengths, avg = genetic_algorithm_with_rl(gen_env, agent=agent)
        best2.append(best_value)
        if best_value == max(best2) and train_flag:
            print('successfully saved!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            agent.save_q_table('q_table.pkl')
    # best_individual=[1, 3, 9, 2, 4, 10, 1, 5, 9, 1, 6, 10, 1, 7, 11, 0, 8, 9, 3, 2, 2, 1, 3, 3, 2, 4, 4, 0, 5, 4, 5, 1, 0, 0, 1, 5]


    show_best(best_individual,env)
    print('best2:', best2)

