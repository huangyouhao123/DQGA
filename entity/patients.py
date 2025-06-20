from entity.resource import Doctor


class Patient:
    def __init__(self, idx, Type, before_cost, OR_cost, after_cost, doctor_idx):
        self.idx = idx
        self.type = Type
        self.doctor_idx = doctor_idx
        self.doctor = None
        self.before_cost = before_cost
        self.OR_cost = OR_cost
        self.after_cost = after_cost
        self.stage = 0
        self.idx_list = None
        self.stage_idx = None
        self.cost = 0
        self.start = 0
        self.end = 0
        self.begin = None
        self.start_time = []
        self.finish_time = []
        self.on = []

    def change_state(self):
        if self.stage == 0:
            self.stage = 1
            self.stage_idx = self.idx_list[ord(self.type) - ord('A') + 1]
            self.cost += self.before_cost
        elif self.stage == 1:
            self.stage = 2
            self.stage_idx = self.idx_list[-1]
            self.cost += self.OR_cost
        else:
            self.stage = 3
            self.cost += self.after_cost
            self.stage_idx = 'a'

    def next_stage_idx(self):
        if self.stage == 0:
            return self.idx_list[ord(self.type) - ord('A') + 1]
        elif self.stage == 1:
            return self.idx_list[-1]
        else:
            raise IndexError('stage index out of range')

    def update(self, s, e, resource_idx):
        self.doctor: Doctor
        if self.stage == 1:
            if self.doctor.end < e:
                self.doctor.end = e
            self.doctor.start.append(s)
            self.doctor.finish.append(e)
            self.doctor._on.insert(self.doctor.start.index(s), self.idx)
        if self.stage == 0:
            self.begin = s
        self.change_state()
        self.end = e
        self.start = s
        self.start_time.append(self.start)
        self.finish_time.append(self.end)
        self.on.append(resource_idx)

    def is_finished(self):
        if self.stage == 3:
            return True
        else:
            return False
