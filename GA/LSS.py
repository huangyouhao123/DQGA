import random
from util.config import config


def mutate(individual):

    mute_list=[change,InsertOperator,SwapOperator,ReverseOperator,ReverseInsertionOperator]
    idx=random.randint(0,len(mute_list)-1)
    return mute_list[idx](individual)


def change(individual):
    max_gene = config.patient_choice_num * 2
    i = random.sample(range(len(individual)), 1)
    i = i[0]
    individual[i] = random.randint(0, max_gene - 1)


def InsertOperator(p: list):
    length = len(p)
    r1 = random.randint(1, length - 1)
    r2 = random.randint(1, length - 1)
    if r2 >= r1:
        element = p.pop(r2)
        p.insert(r1, element)
    else:
        element = p.pop(r2)
        p.insert(r1 - 1, element)
    return p


def SwapOperator(p: list):
    length = len(p)
    r1 = random.randint(1, length - 1)
    r2 = random.randint(1, length - 1)
    p[r1], p[r2] = p[r2], p[r1]
    return p


def ReverseOperator(p):
    length = len(p)
    # 生成两个随机位置
    r1 = random.randint(0, length - 1)
    r2 = random.randint(0, length - 1)

    # 确保r1和r2不相同
    while r1 == r2:
        r2 = random.randint(0, length - 1)

    # 确保r1小于r2
    if r1 > r2:
        r1, r2 = r2, r1

    # 颠倒r1和r2之间的元素
    p[r1:r2 + 1] = p[r1:r2 + 1][::-1]

    return p


def ReverseInsertionOperator(p):
    length = len(p)
    if length < 2:  # 如果列表长度小于2，则无法执行操作
        return p

    # 生成两个随机位置
    r1 = random.randint(0, length - 2)
    r2 = random.randint(0, length - 2)

    # 确保r1和r2不相同，并且r1和r1+1在列表范围内
    while r1 == r2 or r1 == length - 1:
        r2 = random.randint(0, length - 2)

    # 反转r1和r1+1位置的元素
    temp = p[r1]
    p[r1] = p[r1 + 1]
    p[r1 + 1] = temp

    # 确保r1小于r2
    if r1 > r2:
        r1, r2 = r2, r1

    # 将反转后的元素插入到r2-1和r2位置
    p = p[:r2] + [p[r1], p[r1 + 1]] + p[r2:]

    return p
