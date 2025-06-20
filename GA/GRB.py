# 导入 Gurobi 库
import gurobipy as gp
from gurobipy import GRB, quicksum
import matplotlib

# 选择合适的后端
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from RL_utils.env import Env
from util.util import parse_args

args = parse_args()
# data_patients,data_source=get_data(args)
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
    'before': 2,
    'ORtype': 3,
    'after': 3,
    'A': [2, 0.3, 2, 0.333],
    'B': [2, 0.4, 3, 0.333],
    'C': [2, 0.3, 2, 0.333]
}


def Gantt(resource, env):
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    M = ['red', '#5d9dfc', 'yellow', 'orange', 'green', 'palegoldenrod', 'purple', 'pink', 'Thistle', 'Magenta',
         'SlateBlue', 'RoyalBlue', 'Cyan', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod', 'mediumslateblue',
         'navajowhite', 'navy', 'sandybrown', 'moccasin']
    Job_text = ['P' + str(i + 1) for i in range(200)]
    Machine_text = [machine.type + str(machine.idx) for machine in resource]

    # Create the Gantt chart bars
    for i in range(len(resource)):
        for j in range(len(resource[i].start)):
            if resource[i].finish[j] - resource[i].start[j] != 0:
                plt.barh(i, width=resource[i].finish[j] - resource[i].start[j],
                         height=0.8, left=resource[i].start[j],
                         color=M[env.patients[resource[i]._on[j]].doctor.abs_idx],
                         edgecolor='black')
                plt.text(x=resource[i].start[j] + (resource[i].finish[j] - resource[i].start[j]) / 2 - 0.1,
                         y=i,
                         s=Job_text[resource[i]._on[j]],
                         fontsize=12)

    # Set the y-ticks to the machine texts
    plt.yticks(range(len(resource)), Machine_text)

    plt.show()


# 目标函数部分的
type = [chr(ord('A') + i) for i in range(len(data_source.keys()) - 3)]
eta_s = [data_source[tp][3] for tp in type]
L1 = 0.7  # 示例值
L2 = 0.3  # 示例值
L3 = 0.5

# 每个医生只有一个医生负责，每个科室所有医生负责的病人总数等于该科室的病人数量
# [手术间的数量，科室的权重，科室医生的数量]
# 创建模型
model = gp.Model("hospital_scheduling_simplified")

# 参数定义
p = data_source['ORtype']  # 科室数量（简化为3个科室）

n_s_dict = {chr(i + 65): 0 for i in range(len(data_source.keys()) - 3)}
m_s_or = [data_source[key][0] for key in n_s_dict.keys()]  # 每个科室的手术室数量
# data_source.keys() 返回 data_source 字典的所有键（包括 'before'、'after'、'ORtype' 和科室的代号 'A'、'B'、'C'）
# len(data_source.keys()) - 3 计算除了前面的三个键 'before'、'after'、'ORtype' 之外剩下的科室数量。结果为3（科室 'A'、'B'、'C'）
# chr(i + 65) 将 i 转换为ASCII码的字母（65 对应字符 'A'）。所以会生成键 'A'、'B'、'C'
# n_s_dict 生成的字典为 {'A': 0, 'B': 0, 'C': 0}
# n_s_dict.keys() 返回之前生成的字典 n_s_dict 的键，即科室代号 'A'、'B'、'C'
# data_source[key][0] 从 data_source 中对应科室的第一个元素中提取手术室的数量。例如，data_source['A'][0] 为2，data_source['B'][0] 为2，data_source['C'][0] 为3
# m_s_or 生成的列表为 [2, 2, 3]，表示科室 'A'、'B' 和 'C' 的手术室数量分别为 2、2 和 3

m_phu = data_source['before']  # PHU病床数量
m_pacu = data_source['after']  # PACU病床数量

for patient in data_patients:
    n_s_dict[patient[0]] += 1
n_s = list(n_s_dict.values())  # 每个科室的病人数量

g_s = [data_source[key][2] for key in n_s_dict.keys()]  # 每个科室的医生数量

# 操作时间
# 初始化PHU_duration列表
PHU_duration = []

# 遍历每个科室
for s in range(len(data_source.keys()) - 3):  # 遍历'A', 'B', 'C'三个科室
    symbol = chr(s + 65)  # 获取对应科室符号，65是'A'的ASCII码
    # 遍历病人列表
    for patient in data_patients:
        if patient[0] == symbol:  # 如果病人属于第s个科室
            PHU_duration.append(patient[1])  # 将第二列的PHU时长加入PHU_duration

# 输出结果
# print(PHU_duration)

# 初始化PHU_duration列表
PACU_duration = []

# 遍历每个科室
for s in range(len(data_source.keys()) - 3):  # 遍历'A', 'B', 'C'三个科室
    symbol = chr(s + 65)  # 获取对应科室符号，65是'A'的ASCII码
    # 遍历病人列表
    for patient in data_patients:
        if patient[0] == symbol:  # 如果病人属于第s个科室
            PACU_duration.append(patient[3])  # 将第二列的PHU时长加入PHU_duration

# 输出结果
print(PHU_duration)

# PHU_duration = 0.5
# PACU_duration = 0.5
# OR_durations = [2, 1.5, 3]
# 初始化OR_durations列表，用来存储每个科室每个病人的手术时长（第三列的值）
OR_durations = []

# 遍历每个科室
for s in range(len(data_source.keys()) - 3):  # 遍历科室 'A', 'B', 'C'
    symbol = chr(s + 65)  # 获取科室的符号，'A', 'B', 'C'

    # 初始化存储当前科室病人手术时长的列表
    OR_durations_for_s = []

    # 遍历每个病人，找到属于当前科室的病人
    for patient in data_patients:
        if patient[0] == symbol:  # 如果病人属于当前科室
            # 获取病人的OR时长（即第三列的值）
            OR_duration = patient[2]
            OR_durations_for_s.append(OR_duration)  # 添加到当前科室的OR时长列表

    # 将当前科室的OR时长列表添加到OR_durations
    OR_durations.append(OR_durations_for_s)

# 输出OR_durations
for i, durations in enumerate(OR_durations):
    print(f"科室 {chr(i + 65)} 的 OR_durations: {durations}")

total_actual_patients = sum(n_s)

# 决策变量
ID_io = model.addVars([(i, o) for i in range(total_actual_patients) for o in [0, 2]], vtype=GRB.CONTINUOUS, lb=0, ub=2,
                      name="ID_io")
E_max = model.addVar(vtype=GRB.CONTINUOUS, lb=1e-6, ub=16, name="E_max")
E_max_s = model.addVars(p, vtype=GRB.CONTINUOUS, lb=1e-6, ub=16, name="E_max_s")
Avg_ST_s = model.addVars(p, vtype=GRB.CONTINUOUS, lb=0, name="Avg_ST_s")
S_sri = model.addVars(
    [(s, r, i) for s in range(p) for r in range(m_s_or[s]) for i in range(n_s[s])],
    vtype=GRB.CONTINUOUS, lb=0, name="S_sri"
)
E_sri = model.addVars(
    [(s, r, i) for s in range(p) for r in range(m_s_or[s]) for i in range(n_s[s])],
    vtype=GRB.CONTINUOUS, lb=0, name="E_sri"
)
S_rio = model.addVars(
    [(r, i, o) for o in [0, 2] for r in range([m_phu, m_pacu][o // 2]) for i in range(total_actual_patients)],
    vtype=GRB.CONTINUOUS, lb=0, name="S_rio"
)
E_rio = model.addVars(
    [(r, i, o) for o in [0, 2] for r in range([m_phu, m_pacu][o // 2]) for i in range(total_actual_patients)],
    vtype=GRB.CONTINUOUS, lb=0, name="E_rio"
)
x_sir = model.addVars(
    [(s, i, r) for s in range(p) for i in range(n_s[s]) for r in range(m_s_or[s])],
    vtype=GRB.BINARY, name="x_sir"
)
y_ir = model.addVars(
    [(i, r) for i in range(total_actual_patients) for r in range(m_phu)],
    vtype=GRB.BINARY, name="y_ir"
)
z_ir = model.addVars(
    [(i, r) for i in range(total_actual_patients) for r in range(m_pacu)],
    vtype=GRB.BINARY, name="z_ir"
)
alpha_srij = model.addVars(
    [(s, r, i, j) for s in range(p) for r in range(m_s_or[s]) for i in range(n_s[s]) for j in range(n_s[s]) if i != j],
    vtype=GRB.BINARY, name="alpha_srij"
)
belta_rij = model.addVars(
    [(r, i, j) for r in range(m_phu) for i in range(total_actual_patients) for j in range(total_actual_patients) if
     i != j],

    vtype=GRB.BINARY, name="belta_rij"
)
gamma_rij = model.addVars(
    [(r, i, j) for r in range(m_pacu) for i in range(total_actual_patients) for j in range(total_actual_patients) if
     i != j],

    vtype=GRB.BINARY, name="gamma_rij"
)
B_sgij = model.addVars(
    [(s, g, i, j) for s in range(p) for g in range(g_s[s]) for i in range(n_s[s]) for j in range(n_s[s]) if i != j],
    vtype=GRB.BINARY, name="B_sgij"
)
A = []  # 最终矩阵

# 遍历每个科室
for s in range(p):
    symbol = chr(s + 65)  # 获取科室符号，如'A', 'B', 'C'
    department_patients = []

    # 提取该科室的病人列表
    for patient in data_patients:
        if patient[0] == symbol:
            department_patients.append(patient)

    # 根据病人数量初始化该科室的矩阵
    doctor_responsibilities = []
    num_patients = len(department_patients)

    # 创建矩阵行
    for doctor in range(g_s[s]):  # 假设每个科室有3个医生
        doctor_row = [0] * num_patients  # 初始化行
        for idx, patient in enumerate(department_patients):
            if patient[4] == doctor:  # 检查医生是否负责该病人
                doctor_row[idx] = 1  # 如果负责则设置为1
        doctor_responsibilities.append(doctor_row)

    # 将该科室的医生责任矩阵添加到A中
    A.append(doctor_responsibilities)

# 打印最终的A矩阵
print("矩阵 A:")
for department in A:
    print(department)

# 12. w_s: 科室s进行手术的物资成本，满足 sum(w_s) = 1
w_s = [data_source[key][1] for key in n_s_dict.keys()]
# 3. alpha_s: 科室 s 的等待奖励
alpha_s = model.addVars(p, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="alpha_s")
# 11. SLA_s: 科室s的一天累计工作时长
# 初始化SLA_s列表
SLA_s = []

# 遍历每个科室
for s in range(len(data_source.keys()) - 3):  # 遍历三个科室 'A', 'B', 'C'
    symbol = chr(s + 65)  # 获取对应科室的符号，'A', 'B', 'C'
    total_time = 0  # 初始化总时间为0

    # 遍历病人列表
    for patient in data_patients:
        if patient[0] == symbol:  # 如果病人属于当前科室
            total_time += sum(patient[1:4])  # 将第二、三、四列的元素相加

    # 将科室的总时间加入到SLA_s列表中
    SLA_s.append(total_time)

# 输出结果
print(SLA_s)

# # 初始化空矩阵 B_sgij
# B_sgij = {}
# # 遍历科室 s
# for s in range(p):
#     symbol = chr(s + 65)  # 获取科室符号 ('A', 'B', 'C', ...)
#     # 提取当前科室的病人列表（按出现顺序）
#     patients_in_s = [i for i, patient in enumerate(data_patients) if patient[0] == symbol]
#     # 初始化 B[s]
#     B_sgij[s] = {}
#     # 对医生 g 进行循环
#     for g in range(g_s[s]):
#         B_sgij[s][g] = {}
#         # 对病人 i 进行循环
#         for i in range(len(patients_in_s)):
#             B_sgij[s][g][i] = {}
#             # 对病人 j 进行循环
#             for j in range(len(patients_in_s)):
#                 if i < j:
#                     # 获取病人 i 和 j 的医生编号（data_patients 最后一列）
#                     doctor_i = data_patients[patients_in_s[i]][-1]
#                     doctor_j = data_patients[patients_in_s[j]][-1]
#                     # 判断条件，只有当医生编号相同且 i < j 时，B_sgij[s][g][i][j] == 1
#                     if doctor_i == doctor_j:
#                         B_sgij[s][g][i][j] = 1
#                     else:
#                         B_sgij[s][g][i][j] = 0
#                 else:
#                     B_sgij[s][g][i][j] = 0
# # 输出 B_sgij 验证
# for s in B_sgij:
#     print(f"B_sgij for department {chr(s + 65)}:")
#     for g in B_sgij[s]:
#         for i in B_sgij[s][g]:
#             for j in B_sgij[s][g][i]:
#                 print(f"B[{s}][{g}][{i}][{j}] = {B_sgij[s][g][i][j]}")
#


# 约束条件
# 1. 每个病人分配到一个手术室
for s in range(p):
    for i in range(n_s[s]):
        model.addConstr(
            quicksum(x_sir[s, i, r] for r in range(m_s_or[s])) == 1,
            f"OR_Unique_Assignment_{s}_{i}"
        )

# 2. 每个病人分配到一个PHU病床
for i in range(total_actual_patients):
    model.addConstr(
        quicksum(y_ir[i, r] for r in range(m_phu)) == 1,
        f"PHU_Unique_Assignment_{i}"
    )

# 3. 每个病人分配到一个PACU病床
for i in range(total_actual_patients):
    model.addConstr(
        quicksum(z_ir[i, r] for r in range(m_pacu)) == 1,
        f"PACU_Unique_Assignment_{i}"
    )

# 4. 时间约束：S + duration = E
# OR阶段
for s in range(p):
    for r in range(m_s_or[s]):
        for i in range(n_s[s]):
            model.addConstr(
                S_sri[s, r, i] + OR_durations[s][i] == E_sri[s, r, i],
                f"Time_Equality_OR_{s}_{r}_{i}"
            )

# PHU和PACU阶段
for o in [0, 2]:  # 0: PHU, 2: PACU
    for r in range([m_phu, m_pacu][o // 2]):
        for i in range(total_actual_patients):
            model.addConstr(
                S_rio[r, i, o] + (PHU_duration[i] if o == 0 else PACU_duration[i]) == E_rio[r, i, o],
                f"Time_Equality_{['PHU', 'PACU'][o // 2]}_{r}_{i}_{o}"
            )

# 5. 时间顺序约束：确保在同一资源上的病人操作顺序正确
# PHU阶段
for r in range(m_phu):
    for i in range(total_actual_patients):
        for j in range(total_actual_patients):
            if i != j:
                model.addConstr(
                    belta_rij[r, i, j] + belta_rij[r, j, i] == 1,
                    f"PHU_Order_Symmetry_{r}_{i}_{j}"
                )
                model.addConstr(
                    S_rio[r, j, 0] >= E_rio[r, i, 0] - 100 * (3 - y_ir[i, r] - y_ir[j, r] - belta_rij[r, i, j]),
                    f"PHU_Order_{r}_{i}_{j}"
                )

# PACU阶段
for r in range(m_pacu):
    for i in range(total_actual_patients):
        for j in range(total_actual_patients):
            if i != j:
                model.addConstr(
                    gamma_rij[r, i, j] + gamma_rij[r, j, i] == 1,
                    f"PACU_Order_Symmetry_{r}_{i}_{j}"
                )
                model.addConstr(
                    S_rio[r, j, 2] >= E_rio[r, i, 2] - 100 * (3 - z_ir[i, r] - z_ir[j, r] - gamma_rij[r, i, j]),
                    f"PACU_Order_{r}_{i}_{j}"
                )

# OR阶段
for s in range(p):
    for r in range(m_s_or[s]):
        for i in range(n_s[s]):
            for j in range(n_s[s]):
                if i != j:
                    model.addConstr(
                        alpha_srij[s, r, i, j] + alpha_srij[s, r, j, i] == 1,
                        f"OR_Order_Symmetry_{s}_{r}_{i}_{j}"
                    )
                    model.addConstr(
                        S_sri[s, r, j] >= E_sri[s, r, i] - 100 * (
                                3 - x_sir[s, i, r] - x_sir[s, j, r] - alpha_srij[s, r, i, j]),
                        f"OR_Order_{s}_{r}_{i}_{j}"
                    )

# 6. 阶段之间的衔接约束
# PHU到OR
for s in range(p):
    for i in range(n_s[s]):
        patient_id = sum(n_s[:s]) + i
        for r_or in range(m_s_or[s]):
            for r_phu in range(m_phu):
                model.addConstr(
                    S_sri[s, r_or, i] == ID_io[i, 0] + E_rio[r_phu, patient_id, 0],
                    f"Stage_Transition_PHU_OR_{s}_{i}_{r_or}_{r_phu}"
                )

# OR到PACU
for s in range(p):
    for i in range(n_s[s]):
        patient_id = sum(n_s[:s]) + i
        for r_or in range(m_s_or[s]):
            for r_pacu in range(m_pacu):
                model.addConstr(
                    S_rio[r_pacu, patient_id, 2] == ID_io[i, 2] + E_sri[s, r_or, i],
                    f"Stage_Transition_OR_PACU_{s}_{i}_{r_or}_{r_pacu}"
                )

# 7. 最大完工时间约束
for s in range(p):
    for i in range(n_s[s]):
        patient_id = sum(n_s[:s]) + i
        for r in range(m_pacu):
            model.addConstr(
                E_max_s[s] >= E_rio[r, patient_id, 2],
                f"Max_Work_Time_{s}_{i}"
            )

# 8. 平均等待时间约束（简化计算）第一个阶段的开始时间
for s in range(p):
    total_waiting_time = quicksum(
        S_rio[r, sum(n_s[:s]) + i, 0] * y_ir[sum(n_s[:s]) + i, r]
        for r in range(m_phu)
        for i in range(n_s[s])
    )
    model.addConstr(
        Avg_ST_s[s] == total_waiting_time / n_s[s],
        f"Average_Waiting_Time_{s}"
    )
# ⑨ 等待奖励约束
for s in range(p):
    model.addConstr(
        alpha_s[s] == 1 - Avg_ST_s[s] / SLA_s[s],
        f"Waiting_Reward_{s}"
    )

# # 9. 医生分配约束
# for s in range(p):
#     for g in range(g_s[s]):
#         for r1 in range(m_s_or[s]):
#             for i in range(n_s[s]):
#                 for r2 in range(m_s_or[s]):
#                     for j in range(n_s[s]):
#                         model.addConstr(
#                             S_sri[s, r2, j] >= E_sri[s, r1, i] - 100 * (
#                                 3 - A[s][g][i] - A[s][g][j] - B_sgij[s][g][i][j]),
#                             f"Doctor_Order_{s}_{g}_{i}_{j}_r_{r1}_{r2}"
#                         )

for s in range(p):
    for g in range(g_s[s]):
        for r1 in range(m_s_or[s]):
            for i in range(n_s[s]):
                for r2 in range(m_s_or[s]):
                    for j in range(n_s[s]):
                        if i != j:
                            model.addConstr(
                                B_sgij[s, g, i, j] + B_sgij[s, g, j, i] == 1,
                                f"OR_Order_Symmetry_{s}_{g}_{i}_{j}"
                            )
                            model.addConstr(
                                S_sri[s, r2, j] >= E_sri[s, r1, i] - 100 * (
                                        3 - A[s][g][i] - A[s][g][j] - B_sgij[s, g, i, j]),
                                f"Doctor_Order_{s}_{g}_{i}_{j}_r_{r1}_{r2}"
                            )

for s in range(p):
    model.addConstr(
        E_max >= E_max_s[s],
        f"Max_Work_Time_{s}"  # 使用 s 来生成唯一的名称
    )

# 目标函数（完善版）
# 目标函数包含最大完工时间和平均等待时间，并引入权重系数

# 11. inv_E_max_s 变量，用于目标函数中的分数
inv_E_max_s = model.addVars(p, vtype=GRB.CONTINUOUS, name="inv_E_max_s")
for s in range(p):
    model.addConstr(inv_E_max_s[s] * E_max_s[s] == 1,
                    name=f"Inv_E_max_s_{s}")

# 11. inv_E_max 变量，用于目标函数中的分数
inv_E_max = model.addVar(vtype=GRB.CONTINUOUS, name="inv_E_max")
model.addConstr(inv_E_max * E_max == 1, name=f"Inv_E_max")

# ⑪ 目标函数的设立
model.setObjective(
    # L1 * quicksum(w_s[s] * (SLA_s[s] / n_s[s]) * inv_E_max_s[s] for s in range(p)) + L2 * quicksum(eta_s[s]*alpha_s[s] for s in range(p)) - L3 * quicksum(ID_io[i,o] * inv_E_max for i in range(total_actual_patients) for o in [0,2] ),
    L1 * quicksum(w_s[s] * (SLA_s[s] / n_s[s]) * inv_E_max_s[s] for s in range(p)) + L2 * quicksum(
        eta_s[s] * alpha_s[s] for s in range(p)) - L3 * quicksum(
        ID_io[i, o] * inv_E_max for i in range(total_actual_patients) for o in [0, 2]),
    GRB.MAXIMIZE
)

# 设置非凸参数
model.setParam(GRB.Param.NonConvex, 2)
# # 启发式求解设置
# model.setParam('Heuristics', 0.1)  # 设置启发式权重为10%

# 优化模型
model.optimize()

# 检查模型是否可行
if model.status == GRB.OPTIMAL:
    # 输出目标函数的值
    print(f"最优目标值: {model.objVal}\n")

    # 计算目标函数的两个部分
    part1 = sum(w_s[s] * (SLA_s[s] / (n_s[s] * E_max_s[s].X)) for s in range(p))
    part2 = sum(eta_s[s] * alpha_s[s].X for s in range(p))
    part3 = sum(ID_io[i, o].X / E_max.X for i in range(total_actual_patients) for o in [0, 2])
    haah = [alpha_s[s].X for s in range(p)]
    ID_io = [ID_io[i, o].X for i in range(total_actual_patients) for o in [0, 2]]

    # 输出目标函数的两个部分
    print(f"目标函数第一部分: {part1}")
    print(f"目标函数第二部分: {part2}\n")
    print(f"目标函数第三部分: {part3}\n")
    print(f"目标函数第二部分eta_s: {eta_s}\n")
    print(f"目标函数第二部分alpha_s: {haah}\n")
    print(f"目标函数第二部分SLA_s: {SLA_s}\n")
    print(f"目标函数第三部分ID_io: {ID_io}\n")
    print(f"目标函数第三部分E_max: {E_max.X}\n")

    # # 打印科室的最大完工时间和平均等待时间
    # for s in range(p):
    #     print(f"科室 {s} 的最大完工时间 E_max_s = {E_max_s[s].X}")
    #     print(f"科室 {s} 的平均等待时间 Avg_ST_s = {Avg_ST_s[s].X}\n")
    #
    # # 打印手术室分配结果
    # print("手术室分配 (科室 s, 病人 i, 手术室 r):")
    for s in range(p):
        for i in range(n_s[s]):
            for r in range(m_s_or[s]):
                if x_sir[s, i, r].X > 0.5:
                    # print(f"科室 {s}, 病人 {i}, 手术室 {r}")
                    #
                    # # 使用三重索引访问 S_sri 和 OR_durations
                    # print(f"  病人 {i} 手术阶段的开始时间: {S_sri[s, r, i].X}")
                    # print(f"  病人 {i} 手术阶段的持续时间: {OR_durations[s][i]}")
                    pass

    # 打印PHU分配结果
    print("\nPHU 分配 (病人 i, PHU r):")
    for i in range(total_actual_patients):
        for r in range(m_phu):
            if y_ir[i, r].X > 0.5:
                # print(f"病人 {i}, PHU 病床 {r}")
                #
                # # 假设你有每个病人在PHU阶段的开始时间和持续时间变量 start_time_phu 和 duration_phu
                # print(f"  病人 {i} PHU 阶段的开始时间: {S_rio[r,i,0].X}")
                # print(f"  病人 {i} PHU 阶段的持续时间: {PHU_duration[i]}")
                pass

    # 打印PACU分配结果
    print("\nPACU 分配 (病人 i, PACU r):")
    for i in range(total_actual_patients):
        for r in range(m_pacu):
            if z_ir[i, r].X > 0.5:
                pass
                # print(f"病人 {i}, PACU 病床 {r}")
                #
                # # 假设你有每个病人在PACU阶段的开始时间和持续时间变量 start_time_pacu 和 duration_pacu
                # print(f"  病人 {i} PACU 阶段的开始时间: {S_rio[r,i,2].X}")
                # print(f"  病人 {i} PACU 阶段的持续时间: {PACU_duration[i]}")

    args = parse_args()
    env = Env(args, data_source, data_patients)
    env.init_start_matrix()
    # PHU
    for i in range(total_actual_patients):
        for r in range(m_phu):
            if y_ir[i, r].X > 0.5:
                start = S_rio[r, i, 0].X
                duration = PHU_duration[i]
                resource = env.Before_sources[r]
                resource.start.append(start)
                resource.finish.append(start + duration)
                resource.start.sort()
                resource.finish.sort()
                resource._on.insert(resource.start.index(start), i)
    # PACU
    for i in range(total_actual_patients):
        for r in range(m_pacu):
            if z_ir[i, r].X > 0.5:
                start = S_rio[r, i, 2].X
                duration = PACU_duration[i]
                resource = env.After_sources[r]
                resource.start.append(start)
                resource.finish.append(start + duration)
                resource.start.sort()
                resource.finish.sort()
                resource._on.insert(resource.start.index(start), i)
    # OR
    for s in range(p):
        for i in range(n_s[s]):
            for r in range(m_s_or[s]):
                if x_sir[s, i, r].X > 0.5:
                    start = S_sri[s, r, i].X
                    duration = OR_durations[s][i]
                    resources = env.Before_sources + env.Operation_rooms + env.After_sources
                    resource = resources[env.source_idx[s + 1][r]]
                    resource.start.append(start)
                    resource.finish.append(start + duration)
                    resource.start.sort()
                    resource.finish.sort()
                    idx = i + sum(n_s[:s])
                    resource._on.insert(resource.start.index(start), idx)
    Gantt(env.Before_sources + env.Operation_rooms + env.After_sources, env)

else:
    print("优化未找到最优解。状态代码：", model.status)
    if model.status == GRB.INFEASIBLE:
        print("模型不可行，开始进行不可行性分析...")
        model.computeIIS()
        model.write("model.ilp")
        print("不可行性分析完成。不可行的约束已写入 model.ilp 文件。")
