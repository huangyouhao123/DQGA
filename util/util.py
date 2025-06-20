import argparse
import os
import shutil

from matplotlib import pyplot as plt

from util import global_util

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def parse_args():
    parser = argparse.ArgumentParser("Parse configuration")
    parser.add_argument("--independent", default=True, help="Max time varies from department to department")
    parser.add_argument("--dataset", default='6_2_3_3',help='6_2_2_3, 6_2_3_3,8_2_3_3, 10_3_4_3, 13_3_3_3, 18_4_5_4, 20_5_6_5, 23_10_7_10, 26_10_7_10')
    # for render end

    return parser.parse_args()


def Gantt(Machines,env):
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    M = [
        'lightcoral',  # 浅红色
        '#a9c7fc',  # 浅蓝色（较浅的 #5d9dfc）
        'lightyellow',  # 浅黄色
        'lightsalmon',  # 浅橙色
        'lightgreen',  # 浅绿色
        'white',  # 浅金色
        'plum',  # 浅紫色
        'lightpink',  # 浅粉色
        'lavender',  # 浅薰衣草色
        'orchid',  # 浅洋红色（较浅的 Magenta）
        '#8470ff',  # 浅石蓝色（lightslateblue 对应的 Hex）
        'lightskyblue',  # 浅皇家蓝
        'lightcyan',  # 浅青色
        'aquamarine',  # 浅水色
        'white',  # 花白
        'whitesmoke',  # 幽灵白
        'wheat',  # 浅金麸
        'lightsteelblue',  # 浅中蓝
        'papayawhip',  # 木瓜白
        'lightsteelblue',  # 浅海军蓝
        'peachpuff',  # 沙棕色
        'navajowhite',  # 浅鹿皮
    ]

    Job_text = ['P' + str(i + 1) for i in range(200)]
    Machine_text = [machine.type + str(machine.idx) for machine in Machines]

    # 计算图表的大小：宽度根据任务数量，长度根据机器数量
    num_machines = len(Machines)
    num_jobs = sum([len(machine.start) for machine in Machines])  # 任务数量是每台机器的任务数之和
    width = max(10, num_jobs // 5)  # 动态调整宽度，避免太小
    height = max(6, num_machines // 2)  # 动态调整高度，避免太小

    # 设置图表大小
    plt.figure(figsize=(width, height))

    # 创建甘特图的条形图
    for i in range(len(Machines)):
        for j in range(len(Machines[i].start)):
            task_duration = Machines[i].finish[j] - Machines[i].start[j]
            if task_duration != 0:
                # 调整条形图的宽度，避免过窄
                bar_left = Machines[i].start[j] + 8
                bar_width = max(task_duration, 0.1)  # 最小宽度设为0.1，避免过小

                plt.barh(i, width=bar_width, height=0.8, left=bar_left,
                         color=M[env.patients[Machines[i]._on[j]].doctor.abs_idx], edgecolor='black')

                # 动态调整字体大小
                fontsize = 12 - num_machines // 40  # 根据机器数量调整字体大小

                # 判断是否加粗字体（当任务时间小于0.5时加粗）
                font_weight = 'bold' if task_duration < 0.5 else 'normal'

                # 设置字体位置，避免重叠

                text_x = bar_left + bar_width / 2 - 0.05
                text_y = i
                if Machines[i]._on[j] == 12 or Machines[i]._on[j] ==10:
                    text_x+=0.1

                # 在条形图上添加任务标签
                plt.text(x=text_x, y=text_y, s=Job_text[Machines[i]._on[j]],
                         fontsize=fontsize, ha='center', va='center',
                         color='black', weight=font_weight)

    # 设置 y 轴的刻度为机器文本
    plt.yticks(range(len(Machines)), Machine_text)
    plt.tight_layout(pad=2.0)  # 自动调整布局，并增加整体边距
    plt.xlabel('Time/h', fontsize=14, labelpad=-1)  # 修改X轴字体大小，并调整距离
    plt.ylabel('Resources', fontsize=14, labelpad=-5)  # 修改Y轴字体大小，并调整距离
    plt.savefig("GANTT_6B.svg", format="svg")  # 保存为SVG格式
    plt.show()


def copy_data_folder_to_output(args, override=False):
    """
    拷贝data目录到输出目录下
    :param args:
    :param override: 默认是否覆盖
    :return:
    """
    make_new_dir = True
    if args.test:
        make_new_dir = False
    elif os.path.exists(args.output):
        if args.test:
            make_new_dir = False
        else:
            if override:
                key = "y"
            else:
                key = input("输出目录已经存在，是否覆盖? (y/n)")
            if key == "y":
                make_new_dir = True
                shutil.rmtree(args.output)
            else:
                make_new_dir = False
    if make_new_dir:
        shutil.copytree(os.path.join(global_util.get_project_root(), "data"), os.path.join(args.output, "data"))

    args.data_dir = os.path.join(args.output, "data")


def build_network(args, action_size):
    if args.net == "net2":
        q_local_net = Net2(action_size, args.dueling)
    else:
        q_local_net = NetMLP(input_dim, action_size)
    return q_local_net
