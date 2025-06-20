from util.config import config


class Resource:
    def __init__(self, key):
        self.idx = key
        self.is_empty = True
        self.start = []
        self.finish = []
        self._on = []
        self.end = 0
        self.absolute_idx = None

    def load_num(self):
        return len(self.start)

    def handling(self, patient, pt):
        """ji :第i个工作
           pt: 消耗时间"""
        s = self.insert(patient, pt)
        e = s + pt  # 结束时间=开始时间+消耗时间
        self.start.append(s)
        self.finish.append(e)
        if config.find_gap:
            self.start.sort()  # 时间序列
            self.finish.sort()
        self._on.insert(self.start.index(s), patient.idx)
        if self.end < e:
            self.end = e
        patient.update(s, e, self.absolute_idx)

    def Gap(self):
        Gap = 0
        if not self.start:
            return 0
        else:
            Gap += self.start[0]
            if len(self.start) > 1:
                G = [self.start[i + 1] - self.finish[i] for i in range(0, len(self.start) - 1)]
                return Gap + sum(G)
            return Gap

    def Job_time(self):
        if not self.start:
            return []
        else:
            job_list = [(self.finish[idx] - self.start[idx], self._on[idx]) for idx, _ in enumerate(self.start)]
            return job_list

    def judge_gap(self, t):
        Gap = []
        if self.start == []:
            return Gap
        else:
            if self.start[0] > 0 and self.start[0] > t:
                Gap.append([0, self.start[0]])
            if len(self.start) > 1:
                Gap.extend([[self.finish[i], self.start[i + 1]] for i in range(0, len(self.start) - 1) if
                            self.start[i + 1] - self.finish[i] > 0 and self.start[i + 1] > t])
                return Gap
            return Gap

    """往间隙中插入"""

    def insert(self, patient, pt):
        if patient.stage == 1:
            start = max(patient.end, self.end, patient.doctor.end)
        else:
            start = max(patient.end, self.end)
        if config.find_gap:
            if patient.stage == 1:
                start = max(patient.end, self.end, patient.doctor.end)
                return start
            start = max(patient.end, self.end)
            Gap = self.judge_gap(patient.end)
            if Gap != []:
                for Gi in Gap:
                    if Gi[0] >= patient.end and Gi[1] - Gi[0] >= pt:
                        return Gi[0]
                    elif Gi[0] < patient.end and Gi[1] - patient.end >= pt:
                        return patient.end
        return start


class Before_resource(Resource):
    def __init__(self, key):
        super().__init__(key)
        self.type = 'PHU'


class After_resource(Resource):
    def __init__(self, key):
        super().__init__(key)
        self.type = 'PACU'


class OperationRoom(Resource):
    def __init__(self, type, key):
        super().__init__(key)
        self.type = type


class Doctor(Resource):
    def __init__(self, type, key):
        super().__init__(key)
        self.type = type
        self.abs_idx = None

    def set_abs_idx(self, abs_idx):
        self.abs_idx = abs_idx
