from util.config import config


def Dispatch_rule(a, env,forpt=False):
    if forpt:
        pt=a
        return pt,select_source(env,0,pt)

    if a==40: # FCFS
        pt=early_pt(env)
        return pt,select_source(env,0,pt)
    if a==41:
        pt=SPT(env)
        return pt,select_source(env,0,pt)

    patient_choose = config.patient_choice_num
    sc = int(a / patient_choose)
    a = a % patient_choose

    # rule_list = [SPT, LPT, SRM, SRPT, SSO, LSO, LRM, LRPT, LPT_LSO, LPT_TWK, LPT_TWKR,
    #              SPT_SSO, SPT_TWK, SPT_TWKR, LPT_or_TWK, SPT_or_TWK, LPT_or_TWKR, SPT_or_TWKR, ]
    rule_list_short=[SPT,SRM,SRPT,SSO,SPT_SSO,SPT_TWK,SPT_TWKR,SPT_or_TWK,SPT_or_TWKR]
    rule_list_long=[LPT,LSO,LRM,LRPT,LPT_LSO,LPT_TWK,LPT_TWKR,LPT_or_TWK,LPT_or_TWKR]
    rule_list=rule_list_short+rule_list_long
    # rule_list = rule_list_short
    pt = rule_list[a](env)
    return pt, select_source(env, sc, pt)


def select_source(env, sc, pt):
    if sc == 0:
        return ES(env, pt)
    elif sc == 1:
        return SLS(env, pt)


def ES(env, pt):
    """最早可用"""
    patient = env.patients[pt]
    if patient.stage == 0:
        source = env.Before_sources
    elif patient.stage == 1:
        source = []
        for room in env.Operation_rooms:
            if room.type == patient.type:
                source.append(room)
    else:
        source = env.After_sources
    end_list = [sc.end for sc in source]
    idx = end_list.index(min(end_list))
    return source[idx]


def SLS(env, pt):
    """最小负载"""
    patient = env.patients[pt]
    if patient.stage == 0:
        source = env.Before_sources
    elif patient.stage == 1:
        source = []
        for room in env.Operation_rooms:
            if room.type == patient.type:
                source.append(room)
    else:
        source = env.After_sources
    end_list = [sc.load_num() for sc in source]
    idx = end_list.index(min(end_list))
    return source[idx]


def EL_S(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        if patient.stage <= 2:
            pt.append(patient.end)
        else:
            pt.append(float('inf'))
    return pt.index(min(pt))


# select the job with the shortest processing time
def SPT(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        try:
            pt.append(env.start_Matrix[idx, patient.stage_idx].sum())
        except:
            pt.append(float("inf"))
    return pt.index(min(pt))


# select the job with the longest processing time
def LPT(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        try:
            pt.append(env.start_Matrix[idx, patient.stage_idx].sum())
        except:
            pt.append(-1)
    return pt.index(max(pt))


# Select the job with maximum sum of the processing time of the current and subsequent operation
def LPT_LSO(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        try:
            pt.append(
                env.start_Matrix[idx, patient.stage_idx].sum() + env.start_Matrix[idx, patient.next_stage_idx()].sum())
        except:
            try:
                pt.append(env.start_Matrix[idx, patient.stage_idx].sum())
            except:
                pt.append(-1)
    return pt.index(max(pt))


# select the job with the maximum product of current processing time and total working time
def LPT_TWK(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        try:
            pt.append(env.start_Matrix[idx, patient.stage_idx].sum() * env.start_Matrix[idx].sum())
        except:
            pt.append(-1)
    return pt.index(max(pt))


# select the job with the maximum ratio of current processing time to total work time
def LPT_or_TWK(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        try:
            pt.append(env.start_Matrix[idx, patient.stage_idx].sum() / env.start_Matrix[idx].sum())
        except:
            pt.append(-1)
    return pt.index(max(pt))


# select the job with the maximum ratio of current processing time to total working time remaining.
def LPT_or_TWKR(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        fenmu = env.start_Matrix[idx].sum() - patient.cost
        try:
            pt.append(env.start_Matrix[idx, patient.stage_idx].sum() / fenmu)
        except:
            pt.append(-1)
    return pt.index(max(pt))


# select the job with the maximum product of current processing time and total remaining
def LPT_TWKR(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        try:
            pt.append(env.start_Matrix[idx, patient.stage_idx].sum() * (env.start_Matrix[idx].sum() - patient.cost))
        except:
            pt.append(-1)
    return pt.index(max(pt))


# select the job with the longest remaining machining time not include current operation processing time
def LRM(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        try:
            pt.append(env.start_Matrix[idx].sum() - patient.cost - env.start_Matrix[idx, patient.stage_idx].sum())
        except:
            pt.append(-1)
    return pt.index(max(pt))


# select the job with the longest remaining processing time
def LRPT(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        if patient.stage < 3:
            pt.append(env.start_Matrix[idx].sum() - patient.cost)
        else:
            pt.append(-1)
    return pt.index(max(pt))


# select the job with the longest processing time of subsequent operation
def LSO(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        try:
            pt.append(env.start_Matrix[idx, patient.next_stage_idx()].sum())
        except:
            if patient.stage == 2:
                pt.append(0)
            else:
                pt.append(-1)
    return pt.index(max(pt))


# select the job with minimum sum of the processing time of the current and subsequent operation
def SPT_SSO(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        try:
            pt.append(
                env.start_Matrix[idx, patient.stage_idx].sum() + env.start_Matrix[idx, patient.next_stage_idx()].sum())
        except:
            try:
                pt.append(env.start_Matrix[idx, patient.stage_idx].sum())
            except:
                pt.append(float("inf"))
    return pt.index(min(pt))


def SPT_TWK(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        try:
            pt.append(env.start_Matrix[idx, patient.stage_idx].sum() * env.start_Matrix[idx].sum())
        except:
            pt.append(float("inf"))
    return pt.index(min(pt))


def SPT_or_TWK(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        try:
            pt.append(env.start_Matrix[idx, patient.stage_idx].sum() / env.start_Matrix[idx].sum())
        except:
            pt.append(float("inf"))
    return pt.index(min(pt))


def SPT_TWKR(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        try:
            pt.append(env.start_Matrix[idx, patient.stage_idx].sum() * (env.start_Matrix[idx].sum() - patient.cost))
        except:
            pt.append(float("inf"))
    return pt.index(min(pt))


def SPT_or_TWKR(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        fenmu = (env.start_Matrix[idx].sum() - patient.cost)
        try:
            pt.append(env.start_Matrix[idx, patient.stage_idx].sum() / fenmu)
        except:
            pt.append(float("inf"))
    return pt.index(min(pt))


def SRM(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        try:
            pt.append(env.start_Matrix[idx].sum() - patient.cost - env.start_Matrix[idx, patient.stage_idx].sum())
        except:
            pt.append(float("inf"))
    return pt.index(min(pt))


def SRPT(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        if patient.stage < 3:
            pt.append(env.start_Matrix[idx].sum() - patient.cost)
        else:
            pt.append(float("inf"))
    return pt.index(min(pt))


def SSO(env):
    pt = []
    for idx, patient in enumerate(env.patients):
        try:
            pt.append(env.start_Matrix[idx, patient.next_stage_idx()].sum())
        except:
            if patient.stage == 2:
                pt.append(0)
            else:
                pt.append(float("inf"))
    return pt.index(min(pt))


def early_pt(env):
    for idx, patient in enumerate(env.patients):
        if not patient.is_finished():
            return idx
